//! Python entry points for the geo-design engine (power, diagnostics, selection).

use numpy::PyReadonlyArray2;
use panelkit_geo::selection::{select_markets, SelectConfig};
use panelkit_geo::types::Method;
use panelkit_geo::{diagnostics, power_curve};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::convert::mat_from_numpy;
use crate::results::{PyGeoDiagnostics, PyMarketCandidate, PyPowerResult};

fn parse_method(s: &str) -> PyResult<Method> {
    match s.to_lowercase().as_str() {
        "sc" => Ok(Method::Sc),
        "asc" => Ok(Method::Asc),
        "sdid" => Ok(Method::Sdid),
        other => Err(PyValueError::new_err(format!(
            "unknown method '{other}' (expected sc/asc/sdid)"
        ))),
    }
}

/// Power analysis for one method via historical placebo with injected lift.
#[pyfunction]
#[pyo3(signature = (y, treated, test_len, lifts, method="sdid", alpha=0.1, target_power=0.8, min_pre=0))]
#[allow(clippy::too_many_arguments)]
pub fn geo_power(
    py: Python<'_>,
    y: PyReadonlyArray2<f64>,
    treated: Vec<usize>,
    test_len: usize,
    lifts: Vec<f64>,
    method: &str,
    alpha: f64,
    target_power: f64,
    min_pre: usize,
) -> PyResult<PyPowerResult> {
    let m = parse_method(method)?;
    let mat = mat_from_numpy(&y);
    let min_pre = if min_pre == 0 {
        test_len.max(2)
    } else {
        min_pre
    };
    let pr = py.allow_threads(move || {
        power_curve(
            &mat,
            &treated,
            test_len,
            &lifts,
            m,
            alpha,
            target_power,
            min_pre,
        )
    });
    Ok(PyPowerResult {
        method: pr.method.name().to_string(),
        lifts: pr.points.iter().map(|p| p.lift_pct).collect(),
        power: pr.points.iter().map(|p| p.power).collect(),
        est_mean: pr.points.iter().map(|p| p.est_pct_mean).collect(),
        est_lo: pr.points.iter().map(|p| p.est_pct_lo).collect(),
        est_hi: pr.points.iter().map(|p| p.est_pct_hi).collect(),
        mde_pct: pr.mde_pct,
        mde_abs_per_period: pr.mde_abs_per_period,
        mde_cumulative: pr.mde_cumulative,
        crit: pr.crit,
        se_null: pr.se_null,
        n_windows: pr.n_windows,
    })
}

/// Design diagnostics for a treated-market set.
#[pyfunction]
#[pyo3(signature = (y, treated, test_len))]
pub fn geo_diagnostics(
    y: PyReadonlyArray2<f64>,
    treated: Vec<usize>,
    test_len: usize,
) -> PyResult<PyGeoDiagnostics> {
    let mat = mat_from_numpy(&y);
    let d = diagnostics(&mat, &treated, test_len);
    Ok(PyGeoDiagnostics {
        holdout_pct: d.holdout_pct,
        pre_fit_rel: d.pre_fit_rel,
        improvement_vs_naive: d.improvement_vs_naive,
        seasonality_strength: d.seasonality_strength,
        stability_score: d.stability_score,
        confidence: d.confidence,
        warnings: d.warnings,
    })
}

/// Search and rank candidate treatment-market sets.
#[pyfunction]
#[pyo3(signature = (y, eligible, max_treated, test_len, target_lift, method="sdid", alpha=0.1, target_power=0.8, min_pre=0, n_candidates=200, seed=0))]
#[allow(clippy::too_many_arguments)]
pub fn geo_select(
    py: Python<'_>,
    y: PyReadonlyArray2<f64>,
    eligible: Vec<usize>,
    max_treated: usize,
    test_len: usize,
    target_lift: f64,
    method: &str,
    alpha: f64,
    target_power: f64,
    min_pre: usize,
    n_candidates: usize,
    seed: u64,
) -> PyResult<Vec<PyMarketCandidate>> {
    let m = parse_method(method)?;
    let mat = mat_from_numpy(&y);
    let min_pre = if min_pre == 0 {
        test_len.max(2)
    } else {
        min_pre
    };
    let cfg = SelectConfig {
        eligible,
        max_treated,
        test_len,
        target_lift,
        method: m,
        alpha,
        target_power,
        min_pre,
        n_candidates,
        seed,
    };
    let ranked = py.allow_threads(move || select_markets(&mat, &cfg));
    Ok(ranked
        .into_iter()
        .map(|c| PyMarketCandidate {
            treated: c.treated,
            power_at_target: c.power_at_target,
            mde_pct: c.mde_pct,
            holdout_pct: c.holdout_pct,
            pre_fit_rel: c.pre_fit_rel,
            stability_score: c.stability_score,
            confidence: c.confidence,
            score: c.score,
        })
        .collect())
}
