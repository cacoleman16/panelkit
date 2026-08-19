//! Demeaned Synthetic Control (Ferman & Pinto 2021).
//!
//! Classic SC matches the treated unit's pre-treatment *levels* with a convex
//! combination of donors. When no convex combination can match — the usual case
//! once outcomes carry unit-specific levels (fixed effects) that lie outside the
//! donor hull, or when the pre-period fit is imperfect because of transitory
//! noise — the weights are chosen partly to close a level gap they cannot close,
//! and the resulting estimator is biased even as the number of pre-periods
//! grows. Ferman & Pinto derive bias bounds for that case and propose the
//! **demeaned** estimator: subtract each unit's own pre-treatment mean first,
//! fit the weights on the demeaned data, and add the treated unit's level back.
//!
//! ```text
//!   ỹ_t = y_t − ȳ^pre,   Z̃_{t,j} = Z_{t,j} − z̄_j^pre
//!   ŵ   = argmin_{w ∈ Δ} Σ_{t < T₀} (ỹ_t − Z̃_{t,·} w)²
//!   ŷ_t^N = ȳ^pre + Σ_j ŵ_j (Z_{t,j} − z̄_j^pre)          for t ≥ T₀
//! ```
//!
//! Equivalently: SC with a free intercept, keeping the simplex constraint on the
//! slope weights. Matching *deviations* rather than levels removes the additive
//! unit effects from the matching problem entirely, so a treated unit whose level
//! sits outside the donor hull costs nothing, and the weights are spent on the
//! part of the series that actually carries the common factors.
//!
//! Relation to the neighbours in this crate: SDID also fits its unit weights
//! with a free intercept, but then applies *time* weights and reports a weighted
//! 2×2 DiD; ASC keeps level-matching and corrects the residual imbalance with a
//! ridge outcome model. This estimator is the minimal change to classic SC —
//! same weights problem, same counterfactual path, on demeaned data.

use crate::panel::Panel;
use crate::result::ScFit;
use panelkit_linalg::ops::norms::nrm2;
use panelkit_linalg::opt::simplex::{sc_weights_bounded, WeightBounds};
use panelkit_linalg::Mat;

/// Configuration for the demeaned (Ferman–Pinto) synthetic control.
#[derive(Clone, Copy, Debug)]
pub struct FpConfig {
    /// Ridge penalty on the weights (0.0 = as in the paper).
    pub ridge: f64,
    /// Per-donor weight bounds (`lo ≤ w_j ≤ hi`); default = plain simplex.
    pub bounds: WeightBounds,
}

impl Default for FpConfig {
    fn default() -> Self {
        FpConfig {
            ridge: 0.0,
            bounds: WeightBounds::default(),
        }
    }
}

/// Fit demeaned SC on a block-treatment panel.
pub fn fit(panel: &Panel, cfg: FpConfig) -> ScFit {
    let t0 = panel
        .common_treat_time()
        .expect("demeaned SC requires a single common treatment time");
    fit_at(panel, t0, cfg)
}

/// Fit demeaned SC treating `t0` as the first post-period.
pub fn fit_at(panel: &Panel, t0: usize, cfg: FpConfig) -> ScFit {
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

/// Fit demeaned SC for explicit treated / donor blocks. `donor_pre` is
/// `T_pre × J`, `donor_post` is `T_post × J`.
pub fn fit_series(
    y_pre: &[f64],
    y_post: &[f64],
    donor_pre: &Mat,
    donor_post: &Mat,
    donor_ids: Vec<usize>,
    cfg: FpConfig,
) -> ScFit {
    let t_pre = donor_pre.rows();
    let t_post = donor_post.rows();
    let j = donor_pre.cols();

    // Pre-treatment means: one per donor, one for the treated series. With no
    // pre-period there is nothing to demean *or* to fit — fall back to zero
    // means so the estimator degrades to "counterfactual = donor average".
    let inv_pre = if t_pre > 0 { 1.0 / t_pre as f64 } else { 0.0 };
    let mut donor_mean = vec![0.0; j];
    for jc in 0..j {
        let mut s = 0.0;
        for t in 0..t_pre {
            s += donor_pre.get(t, jc);
        }
        donor_mean[jc] = s * inv_pre;
    }
    let y_mean = y_pre.iter().sum::<f64>() * inv_pre;

    // Demeaned pre-period blocks.
    let mut z_dm = Mat::zeros(t_pre, j);
    for jc in 0..j {
        for t in 0..t_pre {
            z_dm.set(t, jc, donor_pre.get(t, jc) - donor_mean[jc]);
        }
    }
    let y_dm: Vec<f64> = y_pre.iter().map(|v| v - y_mean).collect();

    let w = sc_weights_bounded(&z_dm, &y_dm, cfg.ridge, cfg.bounds).w;

    // Pre-period fit is measured on the demeaned scale: levels are matched by
    // construction, so a level gap must not show up as "imbalance".
    let mut pre_resid = vec![0.0; t_pre];
    for t in 0..t_pre {
        let mut hat = 0.0;
        for jc in 0..j {
            hat += w[jc] * z_dm.get(t, jc);
        }
        pre_resid[t] = y_dm[t] - hat;
    }

    // Counterfactual: treated pre-level + weighted donor *deviations*.
    let mut cf_post = vec![0.0; t_post];
    let mut att_path = vec![0.0; t_post];
    for t in 0..t_post {
        let mut dev = 0.0;
        for jc in 0..j {
            dev += w[jc] * (donor_post.get(t, jc) - donor_mean[jc]);
        }
        cf_post[t] = y_mean + dev;
        att_path[t] = y_post[t] - cf_post[t];
    }
    let att = if att_path.is_empty() {
        0.0
    } else {
        att_path.iter().sum::<f64>() / att_path.len() as f64
    };

    ScFit {
        weights: w,
        donor_ids,
        att_path: att_path.clone(),
        att,
        counterfactual_post: cf_post,
        treated_post: y_post.to_vec(),
        pre_rmspe: nrm2(&pre_resid) / (t_pre.max(1) as f64).sqrt(),
        post_rmspe: nrm2(&att_path) / (t_post.max(1) as f64).sqrt(),
    }
}
