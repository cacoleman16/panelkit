//! The CP-ASC family — Conformal Pooled Augmented Synthetic Control and its
//! variants (novel to this project).
//!
//! Motivation: with several treated units (e.g. a multi-DMA marketing test),
//! fitting one augmented SC per treated unit and then *pooling* the per-unit
//! effects yields a conservative, well-calibrated estimator. The pooling is
//! empirical-Bayes: units that fit poorly (high pre-period MSPE) are
//! down-weighted.
//!
//! Inference is by **conformal block permutation under a null-imposed refit**
//! (Chernozhukov–Wüthrich–Zhu style): to test H₀ of no effect, each treated
//! unit's SC weights are re-estimated on **all T periods** (under H₀ the post
//! periods are just more untreated observations), the resulting residual paths
//! are pooled with weights computed from the same full-sample fit, and the
//! post-period block of that path is compared against all circularly-shifted
//! blocks. Estimating everything symmetrically in time is what makes the
//! blocks (approximately) exchangeable under H₀ — permuting the *main* fit's
//! residuals would mix in-sample pre-period imbalance with out-of-sample
//! post-period prediction error and reject too often exactly when the fit is
//! good.
//!
//! Three pooling targets:
//! - [`PoolMode::Mspe`] (**CP-ASC**): weight unit `d` by `1 / (m_d + median(m))`,
//!   where `m_d` is its pre-period MSPE — the empirical-Bayes shrinkage pool.
//! - [`PoolMode::Stratified`] (**Strat-CP-ASC**): stratify units by size (log
//!   baseline level) into quantile bins, MSPE-pool within each stratum, then
//!   average strata by unit count. This protects against a single extremal large
//!   unit dominating the heterogeneous-effect pool.
//! - [`PoolMode::Cumulative`] (**C-AS-CP-ASC**): weight by baseline size, which
//!   targets the baseline-weighted *cumulative* ATT — the quantity that maps to
//!   total dollar lift, rather than the equal-weighted average effect.

use crate::panel::Panel;
use crate::sc::augmented::{fit_series as asc_fit_series, AscConfig};
use panelkit_linalg::ops::matmul::matvec;
use panelkit_linalg::opt::simplex::sc_weights;
use panelkit_linalg::Mat;

/// Pooling target for the CP-ASC family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolMode {
    /// CP-ASC: empirical-Bayes inverse-MSPE pooling.
    Mspe,
    /// Strat-CP-ASC: stratify by size into `n_strata` bins, pool within, average.
    Stratified { n_strata: usize },
    /// C-AS-CP-ASC: baseline-size-weighted (cumulative-dollar) target.
    Cumulative,
}

/// Configuration for the CP-ASC family.
#[derive(Clone, Copy, Debug)]
pub struct CpascConfig {
    pub asc: AscConfig,
    pub mode: PoolMode,
    /// Block length for the conformal block-permutation test (post-period
    /// blocks). If `None`, uses the full post-period as a single block.
    pub block_len: Option<usize>,
}

impl Default for CpascConfig {
    fn default() -> Self {
        CpascConfig {
            asc: AscConfig::default(),
            mode: PoolMode::Mspe,
            block_len: None,
        }
    }
}

/// Per-treated-unit fit summary.
#[derive(Clone, Debug)]
pub struct UnitFit {
    pub unit: usize,
    /// Per-unit ATT.
    pub att: f64,
    /// Pre-period MSPE (fit quality; drives empirical-Bayes weight).
    pub mspe: f64,
    /// Baseline (mean pre-period level) — the size proxy.
    pub baseline: f64,
    /// Pooling weight assigned to this unit.
    pub weight: f64,
    /// Full residual path (pre then post), length `T`.
    pub residual: Vec<f64>,
}

/// Result of a CP-ASC-family fit.
#[derive(Clone, Debug)]
pub struct CpascFit {
    /// Pooled ATT.
    pub att: f64,
    /// Per-unit fits.
    pub units: Vec<UnitFit>,
    /// Pooled residual path of the **main** fit (Σ_d w_d residual_d, pre-period
    /// SC imbalance then post-period effects), length `T`. Descriptive — the
    /// p-value is computed from `null_residual`, not from this path.
    pub pooled_residual: Vec<f64>,
    /// Pooled residual path of the **null-imposed full-sample refit** (the
    /// exchangeable path the conformal permutation actually tests), length `T`.
    pub null_residual: Vec<f64>,
    /// Conformal block-permutation p-value for H₀: no effect.
    pub p_value: f64,
    /// First post-period index.
    pub t0: usize,
}

/// Fit a CP-ASC-family estimator on a block-treatment panel.
pub fn fit(panel: &Panel, cfg: CpascConfig) -> CpascFit {
    let t0 = panel
        .common_treat_time()
        .expect("CP-ASC requires a single common treatment time");
    fit_at(panel, t0, cfg)
}

/// Fit treating `t0` as the first post-period.
pub fn fit_at(panel: &Panel, t0: usize, cfg: CpascConfig) -> CpascFit {
    let treated = panel.treated_units();
    assert!(!treated.is_empty(), "no treated units");
    let (z0, donor_ids) = panel.donor_pre(t0);
    let (donor_post, _) = panel.donor_post(t0);
    assert!(!donor_ids.is_empty(), "no donor units");
    let t = panel.n_periods();

    // Per-unit ASC fits.
    let mut units: Vec<UnitFit> = Vec::with_capacity(treated.len());
    for &u in &treated {
        let y_pre: Vec<f64> = (0..t0).map(|p| panel.outcome(u, p)).collect();
        let y_post: Vec<f64> = (t0..t).map(|p| panel.outcome(u, p)).collect();
        let fit = asc_fit_series(
            &y_pre,
            &y_post,
            &z0,
            &donor_post,
            donor_ids.clone(),
            cfg.asc,
        );

        // Pre-period residual = y_pre − Z₀ w (the SC imbalance); post = att_path.
        let pre_hat = matvec(&z0, &fit.weights);
        let pre_resid: Vec<f64> = y_pre
            .iter()
            .zip(pre_hat.iter())
            .map(|(a, b)| a - b)
            .collect();
        let mspe = pre_resid.iter().map(|r| r * r).sum::<f64>() / pre_resid.len().max(1) as f64;
        let baseline = y_pre.iter().sum::<f64>() / y_pre.len().max(1) as f64;

        let mut residual = pre_resid;
        residual.extend_from_slice(&fit.att_path);

        units.push(UnitFit {
            unit: u,
            att: fit.att,
            mspe,
            baseline,
            weight: 0.0,
            residual,
        });
    }

    // Assign pooling weights by mode.
    assign_weights(&mut units, cfg.mode);

    // Pooled ATT and residual path.
    let att = units.iter().map(|u| u.weight * u.att).sum();
    let mut pooled_residual = vec![0.0; t];
    for uf in &units {
        for (k, &r) in uf.residual.iter().enumerate() {
            pooled_residual[k] += uf.weight * r;
        }
    }

    // --- Conformal inference under the imposed null (CWZ-style). ---
    // Under H₀ (no effect) every period is untreated, so the SC weights may be
    // estimated on the FULL sample; the resulting residual path treats all T
    // periods symmetrically and its blocks are (approximately) exchangeable.
    // The main fit's residuals are NOT exchangeable — the pre block is the
    // in-sample quantity the weight solve minimizes while the post block is
    // out-of-sample prediction error — which made the old permutation reject
    // ~2× the nominal rate precisely when the model fit well. The pooling
    // weights for the null path are likewise computed from full-sample
    // (time-symmetric) MSPE / baseline.
    let mut z_full = Mat::zeros(t, donor_ids.len());
    for (jc, &d) in donor_ids.iter().enumerate() {
        for p in 0..t {
            z_full.set(p, jc, panel.outcome(d, p));
        }
    }
    let mut null_units: Vec<UnitFit> = Vec::with_capacity(treated.len());
    for &u in &treated {
        let y_full: Vec<f64> = (0..t).map(|p| panel.outcome(u, p)).collect();
        let w = sc_weights(&z_full, &y_full, cfg.asc.sc_ridge).w;
        let fit_full = matvec(&z_full, &w);
        let residual: Vec<f64> = y_full
            .iter()
            .zip(fit_full.iter())
            .map(|(a, b)| a - b)
            .collect();
        let mspe = residual.iter().map(|r| r * r).sum::<f64>() / t.max(1) as f64;
        let baseline = y_full.iter().sum::<f64>() / t.max(1) as f64;
        null_units.push(UnitFit {
            unit: u,
            att: 0.0,
            mspe,
            baseline,
            weight: 0.0,
            residual,
        });
    }
    assign_weights(&mut null_units, cfg.mode);
    let mut null_residual = vec![0.0; t];
    for uf in &null_units {
        for (k, &r) in uf.residual.iter().enumerate() {
            null_residual[k] += uf.weight * r;
        }
    }
    let p_value = conformal_pvalue(&null_residual, t0, cfg.block_len);

    CpascFit {
        att,
        units,
        pooled_residual,
        null_residual,
        p_value,
        t0,
    }
}

/// Median of a slice (copy + sort).
fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Assign per-unit pooling weights according to the mode (weights sum to 1).
fn assign_weights(units: &mut [UnitFit], mode: PoolMode) {
    let n = units.len();
    if n == 0 {
        return;
    }
    match mode {
        PoolMode::Mspe => {
            let mspes: Vec<f64> = units.iter().map(|u| u.mspe).collect();
            let med = median(&mspes).max(f64::MIN_POSITIVE);
            let raw: Vec<f64> = units.iter().map(|u| 1.0 / (u.mspe + med)).collect();
            let total: f64 = raw.iter().sum();
            for (u, r) in units.iter_mut().zip(raw) {
                u.weight = r / total;
            }
        }
        PoolMode::Cumulative => {
            // Weight by baseline size (baseline-weighted cumulative target).
            let total: f64 = units.iter().map(|u| u.baseline.abs()).sum();
            if total > 0.0 {
                for u in units.iter_mut() {
                    u.weight = u.baseline.abs() / total;
                }
            } else {
                for u in units.iter_mut() {
                    u.weight = 1.0 / n as f64;
                }
            }
        }
        PoolMode::Stratified { n_strata } => {
            let n_strata = n_strata.max(1).min(n);
            // Order units by size (log baseline). Sort indices by baseline.
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| units[a].baseline.partial_cmp(&units[b].baseline).unwrap());
            // Assign each unit to a stratum by rank.
            let mut stratum_of = vec![0usize; n];
            for (rank, &idx) in order.iter().enumerate() {
                let s = (rank * n_strata) / n; // 0..n_strata-1
                stratum_of[idx] = s.min(n_strata - 1);
            }
            // Within-stratum MSPE weights, scaled by stratum unit-share.
            for s in 0..n_strata {
                let members: Vec<usize> = (0..n).filter(|&i| stratum_of[i] == s).collect();
                if members.is_empty() {
                    continue;
                }
                let mspes: Vec<f64> = members.iter().map(|&i| units[i].mspe).collect();
                let med = median(&mspes).max(f64::MIN_POSITIVE);
                let raw: Vec<f64> = members
                    .iter()
                    .map(|&i| 1.0 / (units[i].mspe + med))
                    .collect();
                let total: f64 = raw.iter().sum();
                let stratum_share = members.len() as f64 / n as f64;
                for (&i, r) in members.iter().zip(raw) {
                    units[i].weight = stratum_share * r / total;
                }
            }
        }
    }
}

/// Conformal block-permutation p-value (Chernozhukov–Wüthrich–Zhu style).
///
/// Under H₀ of no effect and stationary residuals, the post-period block of the
/// residual path is exchangeable with all circularly-shifted blocks of the same
/// length. The statistic is the mean absolute residual within a block; the
/// p-value is the share of candidate blocks at least as extreme as the actual
/// post-period block.
fn conformal_pvalue(residual: &[f64], t0: usize, block_len: Option<usize>) -> f64 {
    let t = residual.len();
    let t_post = t - t0;
    let blk = block_len.unwrap_or(t_post).clamp(1, t);

    let block_stat = |start: usize| -> f64 {
        let mut s = 0.0;
        for k in 0..blk {
            s += residual[(start + k) % t].abs();
        }
        s / blk as f64
    };

    // Observed statistic on the actual post-period block.
    let observed = block_stat(t0);
    // All circular start positions are candidate placebo blocks.
    let mut n_extreme = 0usize;
    for start in 0..t {
        if block_stat(start) >= observed {
            n_extreme += 1;
        }
    }
    n_extreme as f64 / t as f64
}
