//! Python-facing result objects (`#[pyclass]`).

use pyo3::prelude::*;

/// Result of a synthetic-control-family fit, exposed to Python.
#[pyclass(name = "SCResult")]
#[derive(Clone)]
pub struct PyScResult {
    /// Average post-period ATT.
    #[pyo3(get)]
    pub att: f64,
    /// Per-post-period ATT path.
    #[pyo3(get)]
    pub att_path: Vec<f64>,
    /// Estimated post-period counterfactual.
    #[pyo3(get)]
    pub counterfactual: Vec<f64>,
    /// Observed (aggregated) treated outcome on the post-period.
    #[pyo3(get)]
    pub treated_post: Vec<f64>,
    /// Donor weights.
    #[pyo3(get)]
    pub weights: Vec<f64>,
    /// Donor unit indices matching `weights`.
    #[pyo3(get)]
    pub donor_ids: Vec<usize>,
    /// Pre-period RMSPE (fit quality).
    #[pyo3(get)]
    pub pre_rmspe: f64,
    /// Post-period RMSPE.
    #[pyo3(get)]
    pub post_rmspe: f64,
    /// Placebo p-value, if inference was requested.
    #[pyo3(get)]
    pub p_value: Option<f64>,
    /// Standard error from the resampling distribution, if available.
    #[pyo3(get)]
    pub se: Option<f64>,
    /// Lower/upper confidence bounds, if inference was requested.
    #[pyo3(get)]
    pub ci_lower: Option<f64>,
    #[pyo3(get)]
    pub ci_upper: Option<f64>,
    /// Raw inference distribution (e.g. placebo RMSPE ratios), if available.
    #[pyo3(get)]
    pub inference_distribution: Option<Vec<f64>>,
}

#[pymethods]
impl PyScResult {
    fn __repr__(&self) -> String {
        let p = self
            .p_value
            .map(|v| format!(", p={v:.4}"))
            .unwrap_or_default();
        format!(
            "SCResult(att={:.6}{}, pre_rmspe={:.4}, post_rmspe={:.4}, n_donors={})",
            self.att,
            p,
            self.pre_rmspe,
            self.post_rmspe,
            self.donor_ids.len()
        )
    }
}

/// Result of a difference-in-differences fit, exposed to Python.
#[pyclass(name = "DiDResult")]
#[derive(Clone)]
pub struct PyDidResult {
    /// Overall ATT.
    #[pyo3(get)]
    pub att: f64,
    /// Standard error of the overall ATT (cluster-robust / IF-based).
    #[pyo3(get)]
    pub se: f64,
    /// Event-study relative periods.
    #[pyo3(get)]
    pub event_time: Vec<i64>,
    /// Event-study coefficients matching `event_time`.
    #[pyo3(get)]
    pub event_att: Vec<f64>,
    /// Event-study standard errors matching `event_time`.
    #[pyo3(get)]
    pub event_se: Vec<f64>,
}

#[pymethods]
impl PyDidResult {
    fn __repr__(&self) -> String {
        format!(
            "DiDResult(att={:.6}, se={:.6}, n_event_times={})",
            self.att,
            self.se,
            self.event_time.len()
        )
    }
}
