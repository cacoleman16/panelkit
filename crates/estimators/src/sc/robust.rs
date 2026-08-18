//! Robust Synthetic Control / principal-component regression (Amjad, Shah &
//! Shen 2018; Agarwal, Shah, Shen & Song 2021).
//!
//! The matrix-completion view of synthetic control: the donor panel is assumed
//! to be a low-rank signal plus noise, so **de-noise first, fit second**.
//!
//! 1. Hard-threshold the donor matrix's spectrum: `M̂ = Σ_{i ≤ k} σᵢ uᵢ vᵢᵀ`,
//!    keeping the leading `k` singular values (the retained rank).
//! 2. Regress the treated unit's pre-period path on the *de-noised* donor
//!    pre-block by ordinary least squares — no simplex constraint, no intercept
//!    penalty. The de-noising is the regularizer.
//! 3. Read the counterfactual off the de-noised donor post-block.
//!
//! Unlike SC/ASC/SDID this estimator does **not** constrain the weights: they
//! can be negative and need not sum to one, which is what lets it extrapolate
//! when the treated unit sits outside the donor hull — and what makes it the
//! natural complement to the simplex estimators in an ensemble. It is also why
//! the per-donor weight bounds do not apply to it.
//!
//! Implementation note: with `M̂ = U_k S_k V_kᵀ`, every fitted value lives in the
//! column space of `U_k S_k`, so the fit is done in those `k` coordinates
//! (`α`) and the donor weights recovered as `β = V_k α`. That keeps the least
//! squares problem `T_pre × k` and well-conditioned instead of `T_pre × J` and
//! rank-deficient — the numerically stable way to write the same estimator.

use crate::panel::Panel;
use crate::result::ScFit;
use panelkit_linalg::factor::svd::Svd;
use panelkit_linalg::ops::norms::nrm2;
use panelkit_linalg::solve::lstsq::ols;
use panelkit_linalg::Mat;

/// Configuration for robust (spectrally de-noised) synthetic control.
#[derive(Clone, Copy, Debug)]
pub struct RscConfig {
    /// Retained rank. `None` = data-driven: the smallest rank whose singular
    /// values carry at least [`RscConfig::energy`] of the spectral energy.
    pub rank: Option<usize>,
    /// Spectral-energy target for automatic rank selection, in (0, 1].
    pub energy: f64,
    /// Ridge penalty on the (k-dimensional) regression step. Normally 0 — the
    /// truncation already regularizes.
    pub ridge: f64,
}

impl Default for RscConfig {
    fn default() -> Self {
        RscConfig {
            rank: None,
            // Donor panels of business outcomes are dominated by a handful of
            // factors (level, trend, seasonality); 0.999 of the squared spectrum
            // keeps those and drops the noise floor.
            energy: 0.999,
            ridge: 0.0,
        }
    }
}

/// Fit robust SC on a block-treatment panel.
pub fn fit(panel: &Panel, cfg: RscConfig) -> ScFit {
    let t0 = panel
        .common_treat_time()
        .expect("robust SC requires a single common treatment time");
    fit_at(panel, t0, cfg)
}

/// Number of leading singular values carrying `energy` of the squared spectrum,
/// capped at `max_rank` and at least 1.
fn rank_for_energy(s: &[f64], energy: f64, max_rank: usize) -> usize {
    let total: f64 = s.iter().map(|v| v * v).sum();
    if total <= 0.0 || max_rank == 0 {
        return max_rank.max(1);
    }
    let target = energy.clamp(0.0, 1.0) * total;
    let mut acc = 0.0;
    for (i, v) in s.iter().enumerate() {
        acc += v * v;
        if acc >= target || i + 1 >= max_rank {
            return (i + 1).min(max_rank);
        }
    }
    max_rank
}

/// Fit robust SC treating `t0` as the first post-period.
pub fn fit_at(panel: &Panel, t0: usize, cfg: RscConfig) -> ScFit {
    let treated = panel.treated_units();
    assert!(!treated.is_empty(), "no treated units");
    let (donor_pre, donor_ids) = panel.donor_pre(t0);
    assert!(!donor_ids.is_empty(), "no donor (never-treated) units");
    let (donor_post, _) = panel.donor_post(t0);

    let treated_mean = panel.unit_mean(&treated);
    let y_pre: Vec<f64> = treated_mean[..t0].to_vec();
    let y_post: Vec<f64> = treated_mean[t0..].to_vec();

    fit_series(&y_pre, &y_post, &donor_pre, &donor_post, donor_ids, cfg)
}

/// Fit robust SC for explicit treated / donor blocks. `donor_pre` is
/// `T_pre × J`, `donor_post` is `T_post × J`.
pub fn fit_series(
    y_pre: &[f64],
    y_post: &[f64],
    donor_pre: &Mat,
    donor_post: &Mat,
    donor_ids: Vec<usize>,
    cfg: RscConfig,
) -> ScFit {
    let t_pre = donor_pre.rows();
    let t_post = donor_post.rows();
    let j = donor_pre.cols();

    // De-noise the donor panel over ALL periods at once: the post-period donor
    // rows are never-treated data too, and using them sharpens the factor
    // estimates the counterfactual is read off.
    let mut m = Mat::zeros(t_pre + t_post, j);
    for jc in 0..j {
        for t in 0..t_pre {
            m.set(t, jc, donor_pre.get(t, jc));
        }
        for t in 0..t_post {
            m.set(t_pre + t, jc, donor_post.get(t, jc));
        }
    }
    let svd = Svd::new(&m);
    let s = svd.singular_values();
    // The regression has T_pre observations, so more than T_pre factors cannot
    // be identified; keep the design tall.
    let max_rank = s.len().min(t_pre).min(j).max(1);
    let k = match cfg.rank {
        Some(r) => r.clamp(1, max_rank),
        None => rank_for_energy(s, cfg.energy, max_rank),
    };

    // Factor scores A = U_k S_k, split into pre/post rows.
    let u = svd.u();
    let mut a_pre = Mat::zeros(t_pre, k);
    let mut a_post = Mat::zeros(t_post, k);
    for c in 0..k {
        for t in 0..t_pre {
            a_pre.set(t, c, u.get(t, c) * s[c]);
        }
        for t in 0..t_post {
            a_post.set(t, c, u.get(t_pre + t, c) * s[c]);
        }
    }

    // Fit the treated pre-period on the retained factors. A degenerate design
    // (all-zero spectrum) leaves α = 0, i.e. a flat zero counterfactual — the
    // honest answer when the donor pool carries no signal at all.
    let alpha = if cfg.ridge > 0.0 {
        panelkit_linalg::solve::lstsq::ridge(&a_pre, y_pre, cfg.ridge)
            .map(|f| f.coef)
            .unwrap_or_else(|_| vec![0.0; k])
    } else {
        ols(&a_pre, y_pre).unwrap_or_else(|_| vec![0.0; k])
    };

    // Donor weights β = V_k α — reported for interpretability; unconstrained.
    let v = svd.v();
    let mut beta = vec![0.0; j];
    for jc in 0..j {
        let mut acc = 0.0;
        for c in 0..k {
            acc += v.get(jc, c) * alpha[c];
        }
        beta[jc] = acc;
    }

    let mut pre_resid = vec![0.0; t_pre];
    for t in 0..t_pre {
        let mut hat = 0.0;
        for c in 0..k {
            hat += a_pre.get(t, c) * alpha[c];
        }
        pre_resid[t] = y_pre[t] - hat;
    }

    let mut cf_post = vec![0.0; t_post];
    let mut att_path = vec![0.0; t_post];
    for t in 0..t_post {
        let mut hat = 0.0;
        for c in 0..k {
            hat += a_post.get(t, c) * alpha[c];
        }
        cf_post[t] = hat;
        att_path[t] = y_post[t] - hat;
    }
    let att = if att_path.is_empty() {
        0.0
    } else {
        att_path.iter().sum::<f64>() / att_path.len() as f64
    };

    ScFit {
        weights: beta,
        donor_ids,
        att_path: att_path.clone(),
        att,
        counterfactual_post: cf_post,
        treated_post: y_post.to_vec(),
        pre_rmspe: nrm2(&pre_resid) / (t_pre.max(1) as f64).sqrt(),
        post_rmspe: nrm2(&att_path) / (t_post.max(1) as f64).sqrt(),
    }
}
