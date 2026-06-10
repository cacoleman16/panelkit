//! Python entry points for the difference-in-differences family.
//!
//! Treatment timing is passed as a per-unit `cohorts` array: `cohorts[i]` is the
//! period at which unit `i` first becomes treated, or any negative value for a
//! never-treated unit.

use numpy::PyReadonlyArray2;
use panelkit_estimators::did::{
    bacon_decompose, fit_callaway_with, fit_sunab, fit_twfe, BaconKind, ControlGroup,
};
use panelkit_estimators::Panel;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::convert::mat_from_numpy;
use crate::results::{PyBaconComponent, PyBaconResult, PyDidResult};
use crate::validate;

/// Validate + build a staggered-adoption panel. `cohorts[i] < 0` means never
/// treated; a cohort at or beyond the last period is **never treated within
/// the sample** and is normalized to never-treated (the same convention as the
/// R `did` package, which recodes `g > max(t)` to the never-treated group).
fn build_panel(y: PyReadonlyArray2<f64>, cohorts: Vec<i64>) -> PyResult<Panel> {
    let (n, t) = validate::check_panel(&y)?;
    if cohorts.len() != n {
        return Err(PyValueError::new_err(format!(
            "treat_start has {} entries but the panel has {n} units",
            cohorts.len()
        )));
    }
    let mat = mat_from_numpy(&y);
    let starts: Vec<Option<usize>> = cohorts
        .into_iter()
        .map(|c| {
            if c < 0 || c as usize >= t {
                None // never treated (within the sample window)
            } else {
                Some(c as usize)
            }
        })
        .collect();
    Ok(Panel::new(mat, starts))
}

/// The estimators below compare cohorts against clean controls; fail with an
/// actionable message instead of a core panic when none exist.
fn require_never_treated(panel: &Panel, estimator: &str, hint: &str) -> PyResult<()> {
    if panel.never_treated_units().is_empty() {
        return Err(PyValueError::new_err(format!(
            "{estimator} needs at least one never-treated unit (treat_start < 0, or at/after \
             the last period) as a control. {hint}"
        )));
    }
    Ok(())
}

/// Two-way fixed-effects DiD with cluster-robust SE.
#[pyfunction]
pub fn fit_twfe_py(y: PyReadonlyArray2<f64>, cohorts: Vec<i64>) -> PyResult<PyDidResult> {
    let panel = build_panel(y, cohorts)?;
    let fit = fit_twfe(&panel);
    Ok(PyDidResult {
        att: fit.att,
        se: fit.se,
        event_time: Vec::new(),
        event_att: Vec::new(),
        event_se: Vec::new(),
    })
}

/// Callaway & Sant'Anna group-time ATTs aggregated to overall + event study.
/// `control` is "never" (never-treated) or "notyet" (not-yet-treated). When
/// `covariates` (an `N×K` array) is given, uses covariate-adjusted (regression-
/// adjustment) ATTs.
#[pyfunction]
#[pyo3(signature = (y, cohorts, control="never", covariates=None))]
pub fn fit_callaway_py(
    y: PyReadonlyArray2<f64>,
    cohorts: Vec<i64>,
    control: &str,
    covariates: Option<PyReadonlyArray2<f64>>,
) -> PyResult<PyDidResult> {
    let cg = match control {
        "never" => ControlGroup::NeverTreated,
        "notyet" | "not_yet_treated" => ControlGroup::NotYetTreated,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown control group '{other}' (expected never/notyet)"
            )))
        }
    };
    let mut panel = build_panel(y, cohorts)?;
    if cg == ControlGroup::NeverTreated {
        require_never_treated(
            &panel,
            "Callaway-Sant'Anna with control='never'",
            "If every unit eventually adopts, pass control='notyet' to compare \
             against not-yet-treated units instead.",
        )?;
    }
    if let Some(cov) = covariates {
        let cov_view = cov.as_array();
        if cov_view.shape()[0] != panel.n_units() {
            return Err(PyValueError::new_err(format!(
                "covariates must have one row per unit ({}); got {}",
                panel.n_units(),
                cov_view.shape()[0]
            )));
        }
        if cov_view.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "covariates contain NaN or inf; provide complete, finite covariates",
            ));
        }
        panel = panel.with_covariates(mat_from_numpy(&cov));
    }
    let cs = fit_callaway_with(&panel, cg);
    Ok(PyDidResult {
        att: cs.overall_att,
        se: cs.overall_se,
        event_time: cs.event_study.iter().map(|e| e.key).collect(),
        event_att: cs.event_study.iter().map(|e| e.att).collect(),
        event_se: cs.event_study.iter().map(|e| e.se).collect(),
    })
}

/// Sun & Abraham interaction-weighted event study.
#[pyfunction]
pub fn fit_sunab_py(y: PyReadonlyArray2<f64>, cohorts: Vec<i64>) -> PyResult<PyDidResult> {
    let panel = build_panel(y, cohorts)?;
    require_never_treated(
        &panel,
        "Sun-Abraham",
        "The interaction-weighted estimator uses never-treated units as the reference group.",
    )?;
    let sa = fit_sunab(&panel);
    Ok(PyDidResult {
        att: sa.overall_att,
        se: sa.overall_se,
        event_time: sa.event_study.iter().map(|e| e.key).collect(),
        event_att: sa.event_study.iter().map(|e| e.att).collect(),
        event_se: sa.event_study.iter().map(|e| e.se).collect(),
    })
}

/// Goodman-Bacon decomposition of the TWFE estimate into 2×2 comparisons.
#[pyfunction]
pub fn bacon_decompose_py(y: PyReadonlyArray2<f64>, cohorts: Vec<i64>) -> PyResult<PyBaconResult> {
    let panel = build_panel(y, cohorts)?;
    let b = bacon_decompose(&panel);
    let components = b
        .components
        .iter()
        .map(|c| PyBaconComponent {
            kind: match c.kind {
                BaconKind::TreatedVsUntreated => "treated_vs_untreated".to_string(),
                BaconKind::EarlierVsLater => "earlier_vs_later".to_string(),
                BaconKind::LaterVsEarlierForbidden => "later_vs_earlier_forbidden".to_string(),
            },
            treated_cohort: c.treated_cohort,
            comparison_cohort: c.comparison_cohort,
            weight: c.weight,
            estimate: c.estimate,
        })
        .collect();
    Ok(PyBaconResult {
        twfe: b.twfe,
        forbidden_weight: b.forbidden_weight,
        components,
    })
}
