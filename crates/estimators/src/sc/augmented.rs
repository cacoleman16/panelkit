//! Augmented Synthetic Control (Ben-Michael, Feller & Rothstein 2021).
//!
//! Plain SC can leave residual pre-treatment imbalance when no convex
//! combination of donors matches the treated unit. ASC corrects for that
//! imbalance with a ridge outcome model: it regresses donors' post-period
//! outcomes on their pre-period outcomes and uses the fitted model to "augment"
//! the SC counterfactual by the pre-period imbalance projected through the
//! ridge coefficients.
//!
//! For each post period `t`:
//! ```text
//!   ŷ_post[t] = donor_post[t,·]·w  +  (y_pre − Z₀ w)ᵀ η_t
//! ```
//! where `w` are the SC weights, `Z₀` the donor pre-period block, and `η_t` is
//! the ridge map from pre to post outcomes, fitted **with an intercept** as in
//! the paper — i.e. on donor-centered data:
//! `η_t = (Z̃₀ Z̃₀ᵀ + λI)⁻¹ Z̃₀ donor_post[t,·]` with `Z̃₀ = Z₀ − z̄₀ 1ᵀ`
//! (each pre-period row centered across donors). The intercept itself cancels
//! in the correction because `Σ w = 1`, but *estimating* η on centered data is
//! what makes the estimator translation-invariant: with the raw (uncentered)
//! Gram, adding a constant to every outcome changes the ATT. When SC fits
//! perfectly (imbalance ≈ 0), ASC = SC.

use crate::panel::Panel;
use crate::result::ScFit;
use panelkit_linalg::factor::cholesky::Cholesky;
use panelkit_linalg::ops::matmul::{matvec, syrk_aat};
use panelkit_linalg::ops::norms::nrm2;
use panelkit_linalg::opt::simplex::sc_weights;
use panelkit_linalg::Mat;

/// Configuration for augmented SC.
#[derive(Clone, Copy, Debug)]
pub struct AscConfig {
    /// Ridge penalty for the SC weight problem (usually 0).
    pub sc_ridge: f64,
    /// Ridge penalty `λ` for the augmentation outcome model. If `None`, picked
    /// automatically as a fraction of the mean spectral scale.
    pub aug_lambda: Option<f64>,
}

impl Default for AscConfig {
    fn default() -> Self {
        AscConfig {
            sc_ridge: 0.0,
            aug_lambda: None,
        }
    }
}

/// Fit augmented SC on a block-treatment panel.
pub fn fit(panel: &Panel, cfg: AscConfig) -> ScFit {
    let t0 = panel
        .common_treat_time()
        .expect("augmented SC requires a single common treatment time");
    fit_at(panel, t0, cfg)
}

/// Fit augmented SC treating `t0` as the first post-period.
pub fn fit_at(panel: &Panel, t0: usize, cfg: AscConfig) -> ScFit {
    let treated = panel.treated_units();
    assert!(!treated.is_empty(), "no treated units");
    let (z0, donor_ids) = panel.donor_pre(t0);
    let (donor_post, _) = panel.donor_post(t0);
    assert!(!donor_ids.is_empty(), "no donor units");

    let treated_mean = panel.unit_mean(&treated);
    let y_pre: Vec<f64> = treated_mean[..t0].to_vec();
    let y_post: Vec<f64> = treated_mean[t0..].to_vec();

    fit_series(&y_pre, &y_post, &z0, &donor_post, donor_ids, cfg)
}

/// Fit augmented SC for an explicit treated series against explicit donor
/// blocks. Used by `fit_at` and by the CP-ASC family (which fits one ASC per
/// treated unit). `z0` is `T_pre × J`, `donor_post` is `T_post × J`.
pub fn fit_series(
    y_pre: &[f64],
    y_post: &[f64],
    z0: &Mat,
    donor_post: &Mat,
    donor_ids: Vec<usize>,
    cfg: AscConfig,
) -> ScFit {
    // 1. SC weights on the pre-period.
    let w = sc_weights(z0, y_pre, cfg.sc_ridge).w;

    // 2. Pre-period imbalance.
    let pre_hat = matvec(z0, &w);
    let imbalance: Vec<f64> = y_pre
        .iter()
        .zip(pre_hat.iter())
        .map(|(a, b)| a - b)
        .collect();

    // 3. Ridge map from donor pre-outcomes to post-outcomes, fitted with an
    //    intercept: center each pre-period row across donors (Z̃₀ = Z₀ − z̄₀1ᵀ)
    //    and form G = Z̃₀ Z̃₀ᵀ (T_pre × T_pre). Centering the target is
    //    unnecessary — Z̃₀·1 = 0, so the donor-mean component of the rhs drops
    //    out automatically. Without this, the estimator is not translation-
    //    invariant (adding a constant to every outcome changes the ATT).
    let t_pre = z0.rows();
    let j_donors = z0.cols();
    let mut zc = z0.clone();
    for i in 0..t_pre {
        let mut mean = 0.0;
        for j in 0..j_donors {
            mean += zc.get(i, j);
        }
        mean /= j_donors.max(1) as f64;
        for j in 0..j_donors {
            zc.set(i, j, zc.get(i, j) - mean);
        }
    }
    let g = syrk_aat(&zc); // T_pre × T_pre, centered Gram
    let lambda = cfg.aug_lambda.unwrap_or_else(|| {
        // Default: 0.1 × mean diagonal of the centered Gram (mean spectral
        // scale of the donor *variation*, invariant to the outcome level).
        let mut tr = 0.0;
        for i in 0..t_pre {
            tr += g.get(i, i);
        }
        0.1 * tr / t_pre.max(1) as f64
    });
    // λ > 0 makes G + λI SPD. A non-positive λ only arises when the donors
    // carry zero cross-sectional variation (centered Gram ≡ 0) — then there is
    // nothing for the outcome model to learn and the augmentation is skipped
    // (ASC = SC).
    let chol = if lambda > 0.0 {
        Some(Cholesky::new_ridge(&g, lambda).expect("centered Gram + λI is SPD for λ > 0"))
    } else {
        None
    };

    // 4. Augmented counterfactual per post period.
    let t_post = donor_post.rows();
    let mut cf_post = vec![0.0; t_post];
    for t in 0..t_post {
        let dpost_row = donor_post.row_copy(t); // length J
                                                // SC part: donor_post[t,·] · w
        let sc_part: f64 = dpost_row.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        // Ridge map: η_t = (Z̃₀Z̃₀ᵀ + λI)⁻¹ Z̃₀ donor_post[t,·]
        let aug = match &chol {
            Some(chol) => {
                let rhs = matvec(&zc, &dpost_row); // T_pre
                let eta = chol.solve_vec(&rhs); // T_pre
                imbalance.iter().zip(eta.iter()).map(|(a, b)| a * b).sum()
            }
            None => 0.0,
        };
        cf_post[t] = sc_part + aug;
    }

    // 5. Effects.
    let att_path: Vec<f64> = y_post
        .iter()
        .zip(cf_post.iter())
        .map(|(a, b)| a - b)
        .collect();
    let att = if att_path.is_empty() {
        0.0
    } else {
        att_path.iter().sum::<f64>() / att_path.len() as f64
    };
    let pre_rmspe = nrm2(&imbalance) / (t_pre.max(1) as f64).sqrt();
    let post_rmspe = nrm2(&att_path) / (t_post.max(1) as f64).sqrt();

    ScFit {
        weights: w,
        donor_ids,
        att_path,
        att,
        counterfactual_post: cf_post,
        treated_post: y_post.to_vec(),
        pre_rmspe,
        post_rmspe,
    }
}
