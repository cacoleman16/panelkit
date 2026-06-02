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

/// Result of a CP-ASC-family fit, exposed to Python.
#[pyclass(name = "CPASCResult")]
#[derive(Clone)]
pub struct PyCpascResult {
    /// Pooled ATT.
    #[pyo3(get)]
    pub att: f64,
    /// Conformal block-permutation p-value.
    #[pyo3(get)]
    pub p_value: f64,
    /// Treated unit indices.
    #[pyo3(get)]
    pub unit_ids: Vec<usize>,
    /// Per-unit ATT.
    #[pyo3(get)]
    pub unit_att: Vec<f64>,
    /// Per-unit pre-period MSPE (fit quality).
    #[pyo3(get)]
    pub unit_mspe: Vec<f64>,
    /// Per-unit pooling weight.
    #[pyo3(get)]
    pub unit_weight: Vec<f64>,
    /// Pooled residual path (length T).
    #[pyo3(get)]
    pub pooled_residual: Vec<f64>,
    /// First post-period index.
    #[pyo3(get)]
    pub t0: usize,
}

#[pymethods]
impl PyCpascResult {
    fn __repr__(&self) -> String {
        format!(
            "CPASCResult(att={:.6}, p={:.4}, n_treated={})",
            self.att,
            self.p_value,
            self.unit_ids.len()
        )
    }
}

/// One 2×2 comparison in a Goodman-Bacon decomposition.
#[pyclass(name = "BaconComponent")]
#[derive(Clone)]
pub struct PyBaconComponent {
    /// "treated_vs_untreated", "earlier_vs_later", or "later_vs_earlier_forbidden".
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub treated_cohort: usize,
    #[pyo3(get)]
    pub comparison_cohort: Option<usize>,
    #[pyo3(get)]
    pub weight: f64,
    #[pyo3(get)]
    pub estimate: f64,
}

#[pymethods]
impl PyBaconComponent {
    fn __repr__(&self) -> String {
        format!(
            "BaconComponent(kind={}, treated={}, comparison={:?}, weight={:.4}, estimate={:.4})",
            self.kind, self.treated_cohort, self.comparison_cohort, self.weight, self.estimate
        )
    }
}

/// Result of a Goodman-Bacon decomposition.
#[pyclass(name = "BaconResult")]
#[derive(Clone)]
pub struct PyBaconResult {
    /// Weighted-average estimate `Σ wᵢ βᵢ` — equals the TWFE coefficient.
    #[pyo3(get)]
    pub twfe: f64,
    /// Total weight on forbidden (later-vs-earlier) comparisons.
    #[pyo3(get)]
    pub forbidden_weight: f64,
    /// All 2×2 comparisons.
    #[pyo3(get)]
    pub components: Vec<PyBaconComponent>,
}

#[pymethods]
impl PyBaconResult {
    fn __repr__(&self) -> String {
        format!(
            "BaconResult(twfe={:.6}, forbidden_weight={:.4}, n_components={})",
            self.twfe,
            self.forbidden_weight,
            self.components.len()
        )
    }
}
