//! Python entry points for the synthetic-control family.

use numpy::{PyArray1, PyReadonlyArray2, PyReadonlyArray3};
use panelkit_estimators::mcnnm::{fit_mcnnm_at, McnnmConfig};
use panelkit_estimators::sc::cpasc::{fit_at as fit_cpasc_at, CpascConfig, PoolMode};
use panelkit_estimators::sc::{
    fit_asc_at, fit_at, fit_sdid_at, sdid_jackknife_loo_atts, AscConfig, ScConfig, SdidConfig,
};
use panelkit_estimators::{Panel, ScFit};
use panelkit_inference::{
    asc_att_many, block_bootstrap_mean, jackknife_se, normal_quantile, percentile_ci, sc_att_many,
    sc_placebo_method, sdid_att_many, stationary_bootstrap_mean, PlaceboResult, ScMethod,
};
use panelkit_linalg::Mat;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::convert::mat_from_numpy;
use crate::results::{PyCpascResult, PyScResult};
use crate::validate;

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
        placebo_atts: None,
    }
}

/// Attach an in-space placebo test's outputs to a result: the RMSPE-ratio
/// p-value plus the ATT-scale SE / CI from the placebo-ATT null. With zero
/// usable placebos (e.g. a single donor) there is no null distribution —
/// everything stays `None` rather than reporting a vacuous p = 1.0.
fn attach_placebo(result: &mut PyScResult, pb: &PlaceboResult, att: f64, level: f64) {
    if pb.placebo_ratios.is_empty() {
        return;
    }
    result.p_value = Some(pb.p_value);
    result.inference_distribution = Some(pb.placebo_ratios.clone());
    // ATT-scale uncertainty comes from the placebo *ATTs* (mean post-period
    // gap of each donor refit as pseudo-treated): under no effect those are
    // null draws of the ATT estimator, in outcome units. The RMSPE ratios are
    // dimensionless test statistics — the right input for the p-value, the
    // wrong one for an SE/CI.
    let null_ci = percentile_ci(0.0, &pb.placebo_atts, level);
    result.se = Some(null_ci.se);
    result.ci_lower = Some(att + null_ci.lower);
    result.ci_upper = Some(att + null_ci.upper);
    result.placebo_atts = Some(pb.placebo_atts.clone());
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
    py: Python<'_>,
    y: PyReadonlyArray2<f64>,
    treated: Vec<usize>,
    treat_time: usize,
    ridge: f64,
    placebo: bool,
    level: f64,
) -> PyResult<PyScResult> {
    let (n, t) = validate::check_panel(&y)?;
    validate::check_treated(&treated, n)?;
    validate::check_treat_time(treat_time, t)?;
    validate::check_nonneg("ridge", ridge)?;
    validate::check_unit_interval("level", level)?;
    let mat = mat_from_numpy(&y);
    // Everything below is pure Rust on the copied panel — release the GIL so
    // a J-donor placebo run can use every core (and other Python threads keep
    // running during a long fit).
    Ok(py.allow_threads(move || {
        let panel = Panel::block(mat, &treated, treat_time);
        let cfg = ScConfig { ridge };
        let fit = fit_at(&panel, treat_time, cfg);
        let mut result = result_from_fit(&fit);
        if placebo {
            let pb = sc_placebo_method(&panel, ScMethod::Sc(cfg));
            attach_placebo(&mut result, &pb, fit.att, level);
        }
        result
    }))
}

/// Fit Augmented Synthetic Control (Ben-Michael et al. 2021). If `placebo` is
/// true, an in-space placebo test (refitting ASC per donor) attaches the
/// p-value and ATT-scale SE / CI.
#[pyfunction]
#[pyo3(signature = (y, treated, treat_time, sc_ridge=0.0, aug_lambda=None, placebo=false, level=0.95))]
#[allow(clippy::too_many_arguments)]
pub fn fit_asc(
    py: Python<'_>,
    y: PyReadonlyArray2<f64>,
    treated: Vec<usize>,
    treat_time: usize,
    sc_ridge: f64,
    aug_lambda: Option<f64>,
    placebo: bool,
    level: f64,
) -> PyResult<PyScResult> {
    let (n, t) = validate::check_panel(&y)?;
    validate::check_treated(&treated, n)?;
    validate::check_treat_time(treat_time, t)?;
    validate::check_nonneg("sc_ridge", sc_ridge)?;
    if let Some(l) = aug_lambda {
        // λ = 0 makes the augmentation Gram singular whenever T_pre > J;
        // require strictly positive (None = automatic).
        validate::check_pos("aug_lambda", l)?;
    }
    validate::check_unit_interval("level", level)?;
    let mat = mat_from_numpy(&y);
    Ok(py.allow_threads(move || {
        let panel = Panel::block(mat, &treated, treat_time);
        let cfg = AscConfig {
            sc_ridge,
            aug_lambda,
        };
        let fit = fit_asc_at(&panel, treat_time, cfg);
        let mut result = result_from_fit(&fit);
        if placebo {
            let pb = sc_placebo_method(&panel, ScMethod::Asc(cfg));
            attach_placebo(&mut result, &pb, fit.att, level);
        }
        result
    }))
}

/// Fit Synthetic Difference-in-Differences (Arkhangelsky et al. 2021).
///
/// `inference` is `"none"`, `"placebo"` (in-space placebo, refitting SDID per
/// donor), or `"jackknife"` (the fixed-weights leave-one-unit-out jackknife of
/// Arkhangelsky et al. / `synthdid`; needs ≥ 2 treated units).
#[pyfunction]
#[pyo3(signature = (y, treated, treat_time, zeta_scale=1.0, inference="none", level=0.95))]
pub fn fit_sdid(
    py: Python<'_>,
    y: PyReadonlyArray2<f64>,
    treated: Vec<usize>,
    treat_time: usize,
    zeta_scale: f64,
    inference: &str,
    level: f64,
) -> PyResult<PyScResult> {
    let (n, t) = validate::check_panel(&y)?;
    validate::check_treated(&treated, n)?;
    validate::check_treat_time(treat_time, t)?;
    validate::check_nonneg("zeta_scale", zeta_scale)?;
    validate::check_unit_interval("level", level)?;
    match inference {
        "none" | "placebo" | "jackknife" => {}
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown SDID inference '{other}' (expected none/placebo/jackknife)"
            )))
        }
    }
    if inference == "jackknife" && treated.len() < 2 {
        return Err(PyValueError::new_err(
            "the SDID jackknife needs >= 2 treated units; use inference='placebo' for a \
             single treated unit",
        ));
    }
    let inference = inference.to_string();
    let mat = mat_from_numpy(&y);
    Ok(py.allow_threads(move || {
        let panel = Panel::block(mat, &treated, treat_time);
        let cfg = SdidConfig { zeta_scale };
        let fit = fit_sdid_at(&panel, treat_time, cfg);
        let mut result = result_from_fit(&fit);
        match inference.as_str() {
            "placebo" => {
                let pb = sc_placebo_method(&panel, ScMethod::Sdid(cfg));
                attach_placebo(&mut result, &pb, fit.att, level);
            }
            "jackknife" => {
                let loo = sdid_jackknife_loo_atts(&panel, treat_time, cfg);
                let se = jackknife_se(&loo);
                let z = normal_quantile(1.0 - (1.0 - level) / 2.0);
                result.se = Some(se);
                result.ci_lower = Some(fit.att - z * se);
                result.ci_upper = Some(fit.att + z * se);
                result.inference_distribution = Some(loo);
            }
            _ => {}
        }
        result
    }))
}

/// Fit Matrix-Completion NNM (Athey et al. 2021). `max_rank`, when set, uses a
/// fast randomized truncated SVD inside SoftImpute (big speedup, low-rank cap).
#[pyfunction]
// `lambda_` (not `lambda`) so it is usable as a Python keyword argument —
// `lambda` is a reserved word in Python.
#[pyo3(signature = (y, treated, treat_time, lambda_=None, max_iter=200, tol=1e-5, seed=0, max_rank=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_mcnnm(
    py: Python<'_>,
    y: PyReadonlyArray2<f64>,
    treated: Vec<usize>,
    treat_time: usize,
    lambda_: Option<f64>,
    max_iter: usize,
    tol: f64,
    seed: u64,
    max_rank: Option<usize>,
) -> PyResult<PyScResult> {
    let (n, t) = validate::check_panel(&y)?;
    validate::check_treated(&treated, n)?;
    validate::check_treat_time(treat_time, t)?;
    if let Some(l) = lambda_ {
        // λ ≤ 0 collapses SoftImpute to a trivial fixed point ("counterfactual
        // = the zero fill"); require strictly positive (None = cross-validated).
        validate::check_pos("lambda_", l)?;
    }
    validate::check_min_count("max_iter", max_iter, 1)?;
    validate::check_pos("tol", tol)?;
    if let Some(r) = max_rank {
        validate::check_min_count("max_rank", r, 1)?;
    }
    let mat = mat_from_numpy(&y);
    // MC-NNM is the heavy estimator (an SVD per SoftImpute iteration) —
    // release the GIL for the whole fit.
    Ok(py.allow_threads(move || {
        let panel = Panel::block(mat, &treated, treat_time);
        let cfg = McnnmConfig {
            lambda: lambda_,
            max_iter,
            tol,
            seed,
            max_rank,
        };
        result_from_fit(&fit_mcnnm_at(&panel, treat_time, cfg))
    }))
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
    if series.is_empty() || series.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("series must be non-empty and finite"));
    }
    validate::check_min_count("block_len", block_len, 1)?;
    validate::check_min_count("n_reps", n_reps, 1)?;
    validate::check_unit_interval("level", level)?;
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
    let (n, t) = validate::check_panel(&y)?;
    validate::check_treated(&treated, n)?;
    validate::check_treat_time(treat_time, t)?;
    validate::check_nonneg("sc_ridge", sc_ridge)?;
    if let Some(l) = aug_lambda {
        validate::check_pos("aug_lambda", l)?;
    }
    validate::check_min_count("n_strata", n_strata, 1)?;
    if let Some(b) = block_len {
        validate::check_min_count("block_len", b, 1)?;
    }
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
        null_residual: fit.null_residual,
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
    if r == 0 || n == 0 || t == 0 {
        return Err(PyValueError::new_err(format!(
            "panel stack must be non-empty; got shape ({r}, {n}, {t})"
        )));
    }
    validate::check_treated(&treated, n)?;
    validate::check_treat_time(treat_time, t)?;

    // Build the panels while the GIL is held (we touch the numpy buffer here),
    // checking finiteness in the same pass.
    let mut panels = Vec::with_capacity(r);
    for rr in 0..r {
        let mut m = Mat::zeros(n, t);
        for i in 0..n {
            for j in 0..t {
                let v = view[[rr, i, j]];
                if !v.is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "panel stack contains a non-finite value at [{rr}, {i}, {j}]; \
                         panelkit requires complete, finite panels"
                    )));
                }
                m.set(i, j, v);
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
