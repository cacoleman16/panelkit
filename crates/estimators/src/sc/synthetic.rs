//! Classic Synthetic Control (Abadie, Diamond & Hainmueller 2010).
//!
//! Fit non-negative donor weights summing to one so that a convex combination
//! of never-treated donors matches the (aggregated) treated unit over the
//! pre-treatment period, then read the treatment effect off the post-period gap.
//!
//! With multiple treated units we follow the common geo-experiment convention of
//! aggregating them into a single treated series (simple mean) before fitting.
//! The weight problem reduces to the simplex-constrained QP solved by
//! [`panelkit_linalg::opt::simplex::sc_weights`].

use crate::panel::Panel;
use crate::result::ScFit;
use panelkit_linalg::ops::matmul::{matvec, matvec_t};
use panelkit_linalg::ops::norms::nrm2;
use panelkit_linalg::opt::simplex::{sc_weights, solve_fw};
use panelkit_linalg::Mat;

/// Configuration for the synthetic-control fit.
#[derive(Clone, Copy, Debug)]
pub struct ScConfig {
    /// Ridge penalty on the weights (0.0 = classic SC). A small value improves
    /// conditioning when donors are collinear.
    pub ridge: f64,
}

impl Default for ScConfig {
    fn default() -> Self {
        ScConfig { ridge: 0.0 }
    }
}

/// Fit synthetic control on a block-treatment panel.
///
/// Panics if the panel has no single common treatment time or no donors.
pub fn fit(panel: &Panel, cfg: ScConfig) -> ScFit {
    let t0 = panel
        .common_treat_time()
        .expect("synthetic control requires a single common treatment time");
    fit_at(panel, t0, cfg)
}

/// Fit synthetic control treating `t0` as the first post-period. Exposed so the
/// placebo engine can reuse the same machinery on relabeled panels.
pub fn fit_at(panel: &Panel, t0: usize, cfg: ScConfig) -> ScFit {
    let treated = panel.treated_units();
    assert!(!treated.is_empty(), "no treated units");
    let (donor_pre, donor_ids) = panel.donor_pre(t0);
    assert!(!donor_ids.is_empty(), "no donor (never-treated) units");
    let (donor_post, _) = panel.donor_post(t0);

    // Aggregated treated series (simple mean across treated units).
    let treated_mean = panel.unit_mean(&treated);
    let y_pre: Vec<f64> = treated_mean[..t0].to_vec();
    let y_post: Vec<f64> = treated_mean[t0..].to_vec();

    // Solve the simplex-constrained weight problem on the pre-period.
    let sol = sc_weights(&donor_pre, &y_pre, cfg.ridge);
    let w = sol.w;

    sc_fit_from_weights(&w, donor_ids, &donor_pre, &donor_post, &y_pre, &y_post)
}

/// Assemble an [`ScFit`] from solved weights and the donor/treated blocks.
pub(crate) fn sc_fit_from_weights(
    w: &[f64],
    donor_ids: Vec<usize>,
    donor_pre: &Mat,
    donor_post: &Mat,
    y_pre: &[f64],
    y_post: &[f64],
) -> ScFit {
    // Pre-period fitted values and RMSPE.
    let pre_hat = matvec(donor_pre, w);
    let pre_resid: Vec<f64> = y_pre
        .iter()
        .zip(pre_hat.iter())
        .map(|(a, b)| a - b)
        .collect();
    let pre_rmspe = rmse(&pre_resid);

    // Post-period counterfactual and ATT path.
    let cf_post = matvec(donor_post, w);
    let att_path: Vec<f64> = y_post
        .iter()
        .zip(cf_post.iter())
        .map(|(a, b)| a - b)
        .collect();
    let post_rmspe = rmse(&att_path);
    let att = if att_path.is_empty() {
        0.0
    } else {
        att_path.iter().sum::<f64>() / att_path.len() as f64
    };

    ScFit {
        weights: w.to_vec(),
        donor_ids,
        att_path,
        att,
        counterfactual_post: cf_post,
        treated_post: y_post.to_vec(),
        pre_rmspe,
        post_rmspe,
    }
}

/// Root-mean-squared value of a residual vector.
fn rmse(r: &[f64]) -> f64 {
    if r.is_empty() {
        return 0.0;
    }
    nrm2(r) / (r.len() as f64).sqrt()
}

/// Fit SC for an arbitrary treated series against an arbitrary donor pool,
/// used by the placebo engine (which relabels a donor as "treated"). The donor
/// blocks already exclude the placebo-treated column.
pub fn fit_series(
    y_pre: &[f64],
    y_post: &[f64],
    donor_pre: &Mat,
    donor_post: &Mat,
    donor_ids: Vec<usize>,
    ridge: f64,
) -> ScFit {
    let gram = panelkit_linalg::ops::matmul::syrk_ata(donor_pre);
    let b = matvec_t(donor_pre, y_pre);
    let sol = solve_fw(&gram, &b, ridge, 5000, 1e-10);
    sc_fit_from_weights(&sol.w, donor_ids, donor_pre, donor_post, y_pre, y_post)
}
