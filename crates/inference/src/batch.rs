//! Batched, parallel fitting for Monte-Carlo / power-analysis / robustness runs.
//!
//! Running an estimator across thousands of simulated panels is embarrassingly
//! parallel. Doing the loop in Rust (rather than a Python `for` loop calling
//! `.fit()` per rep) wins three ways: cross-replication parallelism via rayon,
//! no per-call FFI/GIL overhead, and no Python object churn. Each call returns
//! just the ATTs (the statistic a power curve needs), keeping the result small.

use crate::parallel::par_map_items;
use panelkit_estimators::sc::{
    fit_asc_at, fit_at as sc_fit_at, fit_sdid_at, AscConfig, ScConfig, SdidConfig,
};
use panelkit_estimators::Panel;

/// Fit synthetic control across many panels in parallel; returns one ATT per
/// panel (replication order preserved).
pub fn sc_att_many(panels: Vec<Panel>, t0: usize, cfg: ScConfig) -> Vec<f64> {
    par_map_items(panels, move |p| sc_fit_at(&p, t0, cfg).att)
}

/// Fit augmented SC across many panels in parallel; returns one ATT per panel.
pub fn asc_att_many(panels: Vec<Panel>, t0: usize, cfg: AscConfig) -> Vec<f64> {
    par_map_items(panels, move |p| fit_asc_at(&p, t0, cfg).att)
}

/// Fit SDID across many panels in parallel; returns one ATT per panel.
pub fn sdid_att_many(panels: Vec<Panel>, t0: usize, cfg: SdidConfig) -> Vec<f64> {
    par_map_items(panels, move |p| fit_sdid_at(&p, t0, cfg).att)
}
