//! Synthetic Difference-in-Differences (Arkhangelsky, Athey, Hirshberg, Imbens
//! & Wager 2021).
//!
//! SDID combines unit weights (à la synthetic control) and time weights, then
//! forms a doubly-weighted 2×2 difference-in-differences:
//!
//! 1. **Unit weights** `ω` over control units: match the treated-average
//!    pre-period path, with a ridge penalty `ζ²·T_pre` and a free intercept,
//!    `ω ≥ 0, Σω = 1`.
//! 2. **Time weights** `λ` over pre periods: match each control's post-period
//!    average, with a free intercept, `λ ≥ 0, Σλ = 1`.
//! 3. **Weighted DiD**:
//!    `τ̂ = (ȳ_tr,post − Σ_t λ_t ȳ_tr,t) − Σ_i ω_i (ȳ_i,post − Σ_t λ_t Y_{i,t})`.
//!
//! The ridge `ζ = (N_tr·T_post)^{1/4} · sd(Δ control pre-outcomes)` follows the
//! paper.

use crate::panel::Panel;
use crate::result::ScFit;
use panelkit_linalg::opt::simplex::solve_fw;
use panelkit_linalg::Mat;

/// Configuration for SDID.
#[derive(Clone, Copy, Debug)]
pub struct SdidConfig {
    /// Multiplier on the paper's default unit-weight ridge `ζ`. 1.0 = as in the
    /// paper; 0.0 disables regularization.
    pub zeta_scale: f64,
}

impl Default for SdidConfig {
    fn default() -> Self {
        SdidConfig { zeta_scale: 1.0 }
    }
}

/// Fit SDID on a block-treatment panel.
pub fn fit(panel: &Panel, cfg: SdidConfig) -> ScFit {
    let t0 = panel
        .common_treat_time()
        .expect("SDID requires a single common treatment time");
    fit_at(panel, t0, cfg)
}

/// Solve a simplex QP with a free intercept by concentrating the intercept out
/// (centering the design and target along the fitting axis), then Frank–Wolfe.
/// `design` is `R×K` (R fitting observations, K simplex variables); `target`
/// length `R`. `eta` is the ridge added to the centered Gram.
fn simplex_with_intercept(design: &Mat, target: &[f64], eta: f64) -> Vec<f64> {
    let r = design.rows();
    let k = design.cols();
    if k == 0 {
        return Vec::new();
    }
    // Column means and target mean over the R rows.
    let mut col_mean = vec![0.0; k];
    for j in 0..k {
        let mut s = 0.0;
        for i in 0..r {
            s += design.get(i, j);
        }
        col_mean[j] = s / r as f64;
    }
    let tgt_mean = target.iter().sum::<f64>() / r as f64;

    // Centered design and target.
    let mut dc = Mat::zeros(r, k);
    for j in 0..k {
        for i in 0..r {
            dc.set(i, j, design.get(i, j) - col_mean[j]);
        }
    }
    let tc: Vec<f64> = target.iter().map(|&v| v - tgt_mean).collect();

    let gram = panelkit_linalg::ops::matmul::syrk_ata(&dc);
    let b = panelkit_linalg::ops::matmul::matvec_t(&dc, &tc);
    solve_fw(&gram, &b, eta, 10000, 1e-11).w
}

/// Fit SDID treating `t0` as the first post-period.
pub fn fit_at(panel: &Panel, t0: usize, cfg: SdidConfig) -> ScFit {
    let treated = panel.treated_units();
    let controls = panel.never_treated_units();
    assert!(!treated.is_empty(), "no treated units");
    assert!(!controls.is_empty(), "no control units");

    let t = panel.n_periods();
    let t_pre = t0;
    let t_post = t - t0;
    assert!(
        t_pre >= 1 && t_post >= 1,
        "SDID needs at least one pre- and one post-period (t0 in 1..n_periods)"
    );
    let n_tr = treated.len();

    // Treated-average series.
    let ytr = panel.unit_mean(&treated);

    // Control blocks: pre (T_pre × J) and post (T_post × J), rows = periods.
    let (ctrl_pre, _) = panel.donor_pre(t0); // T_pre × J
    let (ctrl_post, _) = panel.donor_post(t0); // T_post × J
    let j = controls.len();

    // --- Unit weights ω: fit control pre-outcomes to treated-avg pre path. ---
    // design = ctrl_pre (rows = pre periods, cols = control units), target = ytr_pre.
    let ytr_pre: Vec<f64> = ytr[..t_pre].to_vec();
    // ζ = (N_tr·T_post)^{1/4} · sd(first differences of control pre-outcomes).
    let zeta = {
        let mut diffs = Vec::new();
        for jc in 0..j {
            for tt in 1..t_pre {
                diffs.push(ctrl_pre.get(tt, jc) - ctrl_pre.get(tt - 1, jc));
            }
        }
        let sd = std_dev(&diffs);
        let scale = ((n_tr * t_post) as f64).powf(0.25);
        cfg.zeta_scale * scale * sd
    };
    let eta_unit = (zeta * zeta) * t_pre as f64;
    let omega = simplex_with_intercept(&ctrl_pre, &ytr_pre, eta_unit);

    // --- Time weights λ: fit pre-period outcomes to post-avg, across controls. ---
    // design rows = control units, cols = pre periods => transpose of ctrl_pre.
    let design_time = ctrl_pre.transpose(); // J × T_pre
                                            // target = each control's post-period average.
    let mut ctrl_post_avg = vec![0.0; j];
    for jc in 0..j {
        let mut s = 0.0;
        for tt in 0..t_post {
            s += ctrl_post.get(tt, jc);
        }
        ctrl_post_avg[jc] = s / t_post as f64;
    }
    let lambda = simplex_with_intercept(&design_time, &ctrl_post_avg, 0.0);

    // --- Weighted DiD. ---
    // Treated time-weighted pre level and simple post level.
    let ytr_pre_lambda: f64 = (0..t_pre).map(|tt| lambda[tt] * ytr[tt]).sum();
    let ytr_post: f64 = ytr[t_pre..].iter().sum::<f64>() / t_post as f64;

    // Control contribution.
    let mut ctrl_term = 0.0;
    let mut ctrl_pre_lambda = vec![0.0; j];
    for jc in 0..j {
        let pre_l: f64 = (0..t_pre).map(|tt| lambda[tt] * ctrl_pre.get(tt, jc)).sum();
        ctrl_pre_lambda[jc] = pre_l;
        ctrl_term += omega[jc] * (ctrl_post_avg[jc] - pre_l);
    }
    let att = (ytr_post - ytr_pre_lambda) - ctrl_term;

    // Counterfactual & ATT path on the post-period (consistent: mean = att).
    let mut cf_post = vec![0.0; t_post];
    let mut att_path = vec![0.0; t_post];
    for tt in 0..t_post {
        // cf[t] = ytr_pre_lambda + Σ_i ω_i (Y_{i,t} − ctrl_pre_lambda_i)
        let mut adj = 0.0;
        for jc in 0..j {
            adj += omega[jc] * (ctrl_post.get(tt, jc) - ctrl_pre_lambda[jc]);
        }
        cf_post[tt] = ytr_pre_lambda + adj;
        att_path[tt] = ytr[t_pre + tt] - cf_post[tt];
    }

    // Pre-period fit diagnostic: treated-avg vs ω-weighted control on pre-period.
    let mut pre_resid = vec![0.0; t_pre];
    for tt in 0..t_pre {
        let mut wsum = 0.0;
        for jc in 0..j {
            wsum += omega[jc] * ctrl_pre.get(tt, jc);
        }
        pre_resid[tt] = ytr[tt] - wsum;
    }
    // Remove the level (intercept) before reporting RMSPE.
    let pre_mean = pre_resid.iter().sum::<f64>() / t_pre as f64;
    let pre_rmspe = {
        let ss: f64 = pre_resid.iter().map(|r| (r - pre_mean).powi(2)).sum();
        (ss / t_pre as f64).sqrt()
    };
    let post_rmspe = {
        let m = att_path.iter().sum::<f64>() / t_post as f64;
        let ss: f64 = att_path.iter().map(|r| (r - m).powi(2)).sum();
        (ss / t_post as f64).sqrt()
    };

    ScFit {
        weights: omega,
        donor_ids: controls,
        att_path,
        att,
        counterfactual_post: cf_post,
        treated_post: ytr[t_pre..].to_vec(),
        pre_rmspe,
        post_rmspe,
    }
}

fn std_dev(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    var.sqrt()
}
