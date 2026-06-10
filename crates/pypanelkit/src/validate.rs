//! Input validation at the Python boundary.
//!
//! Every `#[pyfunction]` entry point validates its arguments here and raises a
//! `ValueError` *before* handing anything to the Rust core, so no input
//! reachable from Python can trip a core `assert!`/index panic. (Panics now
//! unwind into a catchable `PanicException` rather than aborting the process,
//! but a plain ValueError with an actionable message is the contract.)

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// The panel must be non-empty and fully finite (no NaN / ±inf cells).
pub fn check_panel(y: &PyReadonlyArray2<f64>) -> PyResult<(usize, usize)> {
    let view = y.as_array();
    let (n, t) = (view.shape()[0], view.shape()[1]);
    if n == 0 || t == 0 {
        return Err(PyValueError::new_err(format!(
            "y must be a non-empty N×T panel; got shape ({n}, {t})"
        )));
    }
    if view.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "y contains NaN or inf; panelkit requires a complete, finite panel",
        ));
    }
    Ok((n, t))
}

/// Treated unit indices: at least one, all in `[0, n)`, no duplicates, and at
/// least one unit left over as a donor/control.
pub fn check_treated(treated: &[usize], n: usize) -> PyResult<()> {
    if treated.is_empty() {
        return Err(PyValueError::new_err(
            "`treated` must list at least one treated unit index",
        ));
    }
    let mut seen = vec![false; n];
    for &u in treated {
        if u >= n {
            return Err(PyValueError::new_err(format!(
                "treated unit index {u} out of range [0, {n})"
            )));
        }
        if seen[u] {
            return Err(PyValueError::new_err(format!(
                "treated unit index {u} listed more than once"
            )));
        }
        seen[u] = true;
    }
    if treated.len() >= n {
        return Err(PyValueError::new_err(
            "every unit is treated; need at least one never-treated unit as a donor/control",
        ));
    }
    Ok(())
}

/// Block treatment time: needs ≥1 pre-period and ≥1 post-period.
pub fn check_treat_time(treat_time: usize, t: usize) -> PyResult<()> {
    if !(1..t).contains(&treat_time) {
        return Err(PyValueError::new_err(format!(
            "treat_time {treat_time} must be in [1, {t}) so there is at least one pre- and one post-period"
        )));
    }
    Ok(())
}

/// A probability-like parameter must lie strictly inside (0, 1).
pub fn check_unit_interval(name: &str, v: f64) -> PyResult<()> {
    if !(v.is_finite() && 0.0 < v && v < 1.0) {
        return Err(PyValueError::new_err(format!(
            "{name} must be in (0, 1); got {v}"
        )));
    }
    Ok(())
}

/// A penalty/scale parameter must be finite and non-negative.
pub fn check_nonneg(name: &str, v: f64) -> PyResult<()> {
    if !(v.is_finite() && v >= 0.0) {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and >= 0; got {v}"
        )));
    }
    Ok(())
}

/// A penalty parameter must be finite and strictly positive.
pub fn check_pos(name: &str, v: f64) -> PyResult<()> {
    if !(v.is_finite() && v > 0.0) {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and > 0; got {v}"
        )));
    }
    Ok(())
}

/// A count parameter must be at least `min`.
pub fn check_min_count(name: &str, v: usize, min: usize) -> PyResult<()> {
    if v < min {
        return Err(PyValueError::new_err(format!(
            "{name} must be >= {min}; got {v}"
        )));
    }
    Ok(())
}

/// Geo test window: `test_len` periods of test plus `min_pre` periods of
/// pre-window must fit in the panel (`min_pre = 0` means the engine default,
/// `max(test_len, 2)`). Returns the effective `min_pre` to pass through.
pub fn check_geo_window(test_len: usize, min_pre: usize, t: usize) -> PyResult<usize> {
    if test_len == 0 || test_len >= t {
        return Err(PyValueError::new_err(format!(
            "test_len must be in [1, {t}) (panel has {t} periods); got {test_len}"
        )));
    }
    let effective = if min_pre == 0 {
        test_len.max(2)
    } else {
        min_pre
    };
    if effective > t - test_len {
        let max_len = usable_test_len(t, min_pre);
        return Err(PyValueError::new_err(format!(
            "test_len={test_len} with a {effective}-period pre-window needs at least \
             {} periods, but the panel has {t}. Reduce test_len (max usable here: {}) \
             or pass a smaller min_pre.",
            test_len + effective,
            max_len,
        )));
    }
    Ok(effective)
}

/// Largest `test_len` that still fits `min_pre` (or the default pre-window).
fn usable_test_len(t: usize, min_pre: usize) -> usize {
    (1..t)
        .rev()
        .find(|&l| {
            let eff = if min_pre == 0 { l.max(2) } else { min_pre };
            eff <= t - l
        })
        .unwrap_or(0)
}

/// Lift grid for power analysis: finite and non-negative. (Detection is
/// two-sided on |ATT|, so negative lifts add no information — and they would
/// make "minimum detectable effect" ill-defined.)
pub fn check_lifts(lifts: &[f64]) -> PyResult<()> {
    for &l in lifts {
        if !l.is_finite() || l < 0.0 {
            return Err(PyValueError::new_err(format!(
                "lifts must be finite and >= 0 (detection is two-sided on |ATT|, so a \
                 negative lift is equivalent to its magnitude); got {l}"
            )));
        }
    }
    Ok(())
}

/// Generic index list bound check (eligible / include market lists).
pub fn check_indices(name: &str, idx: &[usize], n: usize) -> PyResult<()> {
    for &u in idx {
        if u >= n {
            return Err(PyValueError::new_err(format!(
                "{name} index {u} out of range [0, {n})"
            )));
        }
    }
    Ok(())
}
