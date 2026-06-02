//! Python bindings for panelkit. This crate is the *only* place that depends on
//! `pyo3`/`numpy`; the numerical core and estimators stay Python-agnostic.

// The #[pyfunction] macro can emit an identity `.into()` on the returned
// PyResult; that's outside our control, so silence it crate-wide.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

mod api_sc;
mod convert;
mod results;

/// The panelkit version string.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The compiled extension module `panelkit._panelkit`.
#[pymodule]
fn _panelkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(api_sc::fit_sc, m)?)?;
    m.add_function(wrap_pyfunction!(api_sc::fit_asc, m)?)?;
    m.add_function(wrap_pyfunction!(api_sc::fit_sdid, m)?)?;
    m.add_function(wrap_pyfunction!(api_sc::fit_mcnnm, m)?)?;
    m.add_class::<results::PyScResult>()?;
    Ok(())
}
