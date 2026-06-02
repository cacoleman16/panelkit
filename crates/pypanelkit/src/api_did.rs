//! Python entry points for the difference-in-differences family.
//!
//! Treatment timing is passed as a per-unit `cohorts` array: `cohorts[i]` is the
//! period at which unit `i` first becomes treated, or any negative value for a
//! never-treated unit.

use numpy::PyReadonlyArray2;
use panelkit_estimators::did::{fit_callaway, fit_sunab, fit_twfe};
use panelkit_estimators::Panel;
use pyo3::prelude::*;

use crate::convert::mat_from_numpy;
use crate::results::PyDidResult;

fn build_panel(y: PyReadonlyArray2<f64>, cohorts: Vec<i64>) -> Panel {
    let mat = mat_from_numpy(&y);
    let starts: Vec<Option<usize>> = cohorts
        .into_iter()
        .map(|c| if c < 0 { None } else { Some(c as usize) })
        .collect();
    Panel::new(mat, starts)
}

/// Two-way fixed-effects DiD with cluster-robust SE.
#[pyfunction]
pub fn fit_twfe_py(y: PyReadonlyArray2<f64>, cohorts: Vec<i64>) -> PyResult<PyDidResult> {
    let panel = build_panel(y, cohorts);
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
#[pyfunction]
pub fn fit_callaway_py(y: PyReadonlyArray2<f64>, cohorts: Vec<i64>) -> PyResult<PyDidResult> {
    let panel = build_panel(y, cohorts);
    let cs = fit_callaway(&panel);
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
    let panel = build_panel(y, cohorts);
    let sa = fit_sunab(&panel);
    Ok(PyDidResult {
        att: sa.overall_att,
        se: sa.overall_se,
        event_time: sa.event_study.iter().map(|e| e.key).collect(),
        event_att: sa.event_study.iter().map(|e| e.att).collect(),
        event_se: sa.event_study.iter().map(|e| e.se).collect(),
    })
}
