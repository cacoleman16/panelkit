//! Python entry points for the synthetic-control family.

use numpy::{PyArray1, PyReadonlyArray2, PyReadonlyArray3};
use panelkit_estimators::mcnnm::{fit_mcnnm_at, McnnmConfig};
use panelkit_estimators::sc::cpasc::{fit_at as fit_cpasc_at, CpascConfig, PoolMode};
use panelkit_estimators::sc::{fit_asc_at, fit_at, fit_sdid_at, AscConfig, ScConfig, SdidConfig};
use panelkit_estimators::{Panel, ScFit};
use panelkit_inference::{
    asc_att_many, block_bootstrap_mean, percentile_ci, sc_att_many, sc_placebo, sdid_att_many,
    stationary_bootstrap_mean,
};
use panelkit_linalg::Mat;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::convert::mat_from_numpy;
use crate::results::{PyCpascResult, PyScResult};

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

/// Fit Matrix-Completion NNM (Athey et al. 2021). `max_rank`, when set, uses a
/// fast randomized truncated SVD inside SoftImpute (big speedup, low-rank cap).
#[pyfunction]
#[pyo3(signature = (y, treated, treat_time, lambda=None, max_iter=200, tol=1e-5, seed=0, max_rank=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_mcnnm(
    y: PyReadonlyArray2<f64>,
    treated: Vec<usize>,
    treat_time: usize,
    lambda: Option<f64>,
    max_iter: usize,
    tol: f64,
    seed: u64,
    max_rank: Option<usize>,
) -> PyResult<PyScResult> {
    let panel = Panel::block(mat_from_numpy(&y), &treated, treat_time);
    let cfg = McnnmConfig {
        lambda,
        max_iter,
        tol,
        seed,
        max_rank,
    };
    Ok(result_from_fit(&fit_mcnnm_at(&panel, treat_time, cfg)))
}

/// Block / stationary bootstrap of the mean of a series (e.g. a post-period
/// gap path). Returns `(se, ci_lower, ci_upper)`. `kind` is "block" or
/// "stationary"; `block_len` is the (mean) block length.
#[pyfunction]
#[pyo3(signature = (series, kind="block", block_len=4, n_reps=2000, seed=0, level=0.95))]
pub fn bootstrap_mean(
    series: Vec<f64>,
    kind: &str,
    block_len: usize,
    n_reps: usize,
    seed: u64,
    level: f64,
) -> PyResult<(f64, f64, f64)> {
    let (ci, _draws) = match kind {
        "block" => block_bootstrap_mean(&series, block_len, n_reps, seed, level),
        "stationary" => stationary_bootstrap_mean(&series, block_len, n_reps, seed, level),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown bootstrap kind '{other}' (expected block/stationary)"
            )))
        }
    };
    Ok((ci.se, ci.lower, ci.upper))
}

/// Fit a CP-ASC-family estimator (novel). `mode` is "mspe" (CP-ASC),
/// "stratified" (Strat-CP-ASC), or "cumulative" (C-AS-CP-ASC).
#[pyfunction]
#[pyo3(signature = (y, treated, treat_time, mode="mspe", n_strata=3, block_len=None, sc_ridge=0.0, aug_lambda=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_cpasc(
    y: PyReadonlyArray2<f64>,
    treated: Vec<usize>,
    treat_time: usize,
    mode: &str,
    n_strata: usize,
    block_len: Option<usize>,
    sc_ridge: f64,
    aug_lambda: Option<f64>,
) -> PyResult<PyCpascResult> {
    let pool = match mode {
        "mspe" => PoolMode::Mspe,
        "stratified" => PoolMode::Stratified { n_strata },
        "cumulative" => PoolMode::Cumulative,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown CP-ASC mode '{other}' (expected mspe/stratified/cumulative)"
            )))
        }
    };
    let panel = Panel::block(mat_from_numpy(&y), &treated, treat_time);
    let cfg = CpascConfig {
        asc: AscConfig {
            sc_ridge,
            aug_lambda,
        },
        mode: pool,
        block_len,
    };
    let fit = fit_cpasc_at(&panel, treat_time, cfg);
    Ok(PyCpascResult {
        att: fit.att,
        p_value: fit.p_value,
        unit_ids: fit.units.iter().map(|u| u.unit).collect(),
        unit_att: fit.units.iter().map(|u| u.att).collect(),
        unit_mspe: fit.units.iter().map(|u| u.mspe).collect(),
        unit_weight: fit.units.iter().map(|u| u.weight).collect(),
        pooled_residual: fit.pooled_residual,
        t0: fit.t0,
    })
}

/// Batched, parallel fitting for Monte-Carlo / power-analysis / robustness runs.
///
/// `y3` is a stack of panels `(R, N, T)` — replication `r` is an `N×T` panel
/// sharing the same `treated`/`treat_time`. Returns an array of `R` ATTs, one
/// per replication, computed in parallel in Rust with the GIL released.
/// `method` is "sc", "asc", or "sdid".
#[pyfunction]
#[pyo3(signature = (y3, treated, treat_time, method="sc", ridge=0.0, zeta_scale=1.0))]
pub fn fit_many<'py>(
    py: Python<'py>,
    y3: PyReadonlyArray3<f64>,
    treated: Vec<usize>,
    treat_time: usize,
    method: &str,
    ridge: f64,
    zeta_scale: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let view = y3.as_array();
    let (r, n, t) = (view.shape()[0], view.shape()[1], view.shape()[2]);

    // Build the panels while the GIL is held (we touch the numpy buffer here).
    let mut panels = Vec::with_capacity(r);
    for rr in 0..r {
        let mut m = Mat::zeros(n, t);
        for i in 0..n {
            for j in 0..t {
                m.set(i, j, view[[rr, i, j]]);
            }
        }
        panels.push(Panel::block(m, &treated, treat_time));
    }

    // The fitting is pure Rust — release the GIL so rayon can use every core.
    let method = method.to_string();
    let atts = py
        .allow_threads(move || match method.as_str() {
            "sc" => Ok(sc_att_many(panels, treat_time, ScConfig { ridge })),
            "asc" => Ok(asc_att_many(
                panels,
                treat_time,
                AscConfig {
                    sc_ridge: ridge,
                    aug_lambda: None,
                },
            )),
            "sdid" => Ok(sdid_att_many(panels, treat_time, SdidConfig { zeta_scale })),
            other => Err(format!("unknown method '{other}' (expected sc/asc/sdid)")),
        })
        .map_err(PyValueError::new_err)?;

    Ok(PyArray1::from_vec(py, atts))
}
