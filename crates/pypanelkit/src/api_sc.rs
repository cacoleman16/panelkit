//! Python entry points for the synthetic-control family.

use numpy::PyReadonlyArray2;
use panelkit_estimators::mcnnm::{fit_mcnnm_at, McnnmConfig};
use panelkit_estimators::sc::{fit_asc_at, fit_at, fit_sdid_at, AscConfig, ScConfig, SdidConfig};
use panelkit_estimators::{Panel, ScFit};
use panelkit_inference::{percentile_ci, sc_placebo};
use pyo3::prelude::*;

use crate::convert::mat_from_numpy;
use crate::results::PyScResult;

/// Assemble a bare [`PyScResult`] (no inference attached) from an [`ScFit`].
fn result_from_fit(fit: &ScFit) -> PyScResult {
    PyScResult {
        att: fit.att,
        att_path: fit.att_path.clone(),
        counterfactual: fit.counterfactual_post.clone(),
        treated_post: fit.treated_post.clone(),
        weights: fit.weights.clone(),
        donor_ids: fit.donor_ids.clone(),
        pre_rmspe: fit.pre_rmspe,
        post_rmspe: fit.post_rmspe,
        p_value: None,
        se: None,
        ci_lower: None,
        ci_upper: None,
        inference_distribution: None,
    }
}

/// Fit synthetic control.
///
/// `y` is an `N×T` outcome array (rows = units, cols = periods). `treated` are
/// the row indices of treated units; `treat_time` is the first post-period
/// column index. If `placebo` is true, an in-space placebo test is run and the
/// p-value / distribution are attached.
#[pyfunction]
#[pyo3(signature = (y, treated, treat_time, ridge=0.0, placebo=false, level=0.95))]
pub fn fit_sc(
    y: PyReadonlyArray2<f64>,
    treated: Vec<usize>,
    treat_time: usize,
    ridge: f64,
    placebo: bool,
    level: f64,
) -> PyResult<PyScResult> {
    let mat = mat_from_numpy(&y);
    let panel = Panel::block(mat, &treated, treat_time);
    let cfg = ScConfig { ridge };

    let fit = fit_at(&panel, treat_time, cfg);

    let mut result = result_from_fit(&fit);

    if placebo {
        let pb = sc_placebo(&panel, cfg);
        result.p_value = Some(pb.p_value);
        result.inference_distribution = Some(pb.placebo_ratios.clone());
        // A placebo-based CI on the ATT: use the spread of per-donor placebo
        // post-period gaps as the null reference. Here we report a percentile
        // interval of the placebo ATT-equivalent (ratio×pre_rmspe) for context.
        let draws: Vec<f64> = pb.placebo_ratios.clone();
        if !draws.is_empty() {
            let ci = percentile_ci(pb.treated_ratio, &draws, level);
            result.se = Some(ci.se);
        }
    }

    Ok(result)
}

/// Fit Augmented Synthetic Control (Ben-Michael et al. 2021).
#[pyfunction]
#[pyo3(signature = (y, treated, treat_time, sc_ridge=0.0, aug_lambda=None))]
pub fn fit_asc(
    y: PyReadonlyArray2<f64>,
    treated: Vec<usize>,
    treat_time: usize,
    sc_ridge: f64,
    aug_lambda: Option<f64>,
) -> PyResult<PyScResult> {
    let panel = Panel::block(mat_from_numpy(&y), &treated, treat_time);
    let cfg = AscConfig {
        sc_ridge,
        aug_lambda,
    };
    Ok(result_from_fit(&fit_asc_at(&panel, treat_time, cfg)))
}

/// Fit Synthetic Difference-in-Differences (Arkhangelsky et al. 2021).
#[pyfunction]
#[pyo3(signature = (y, treated, treat_time, zeta_scale=1.0))]
pub fn fit_sdid(
    y: PyReadonlyArray2<f64>,
    treated: Vec<usize>,
    treat_time: usize,
    zeta_scale: f64,
) -> PyResult<PyScResult> {
    let panel = Panel::block(mat_from_numpy(&y), &treated, treat_time);
    let cfg = SdidConfig { zeta_scale };
    Ok(result_from_fit(&fit_sdid_at(&panel, treat_time, cfg)))
}

/// Fit Matrix-Completion NNM (Athey et al. 2021).
#[pyfunction]
#[pyo3(signature = (y, treated, treat_time, lambda=None, max_iter=200, tol=1e-5, seed=0))]
pub fn fit_mcnnm(
    y: PyReadonlyArray2<f64>,
    treated: Vec<usize>,
    treat_time: usize,
    lambda: Option<f64>,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> PyResult<PyScResult> {
    let panel = Panel::block(mat_from_numpy(&y), &treated, treat_time);
    let cfg = McnnmConfig {
        lambda,
        max_iter,
        tol,
        seed,
    };
    Ok(result_from_fit(&fit_mcnnm_at(&panel, treat_time, cfg)))
}
