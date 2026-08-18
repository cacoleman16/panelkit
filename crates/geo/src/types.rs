//! Shared types for geo-experiment design.

use panelkit_linalg::opt::simplex::WeightBounds;

/// Which estimator to power/evaluate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Synthetic Control.
    Sc,
    /// Augmented Synthetic Control.
    Asc,
    /// Synthetic Difference-in-Differences.
    Sdid,
    /// Demeaned Synthetic Control (Ferman & Pinto 2021).
    Fp,
    /// Robust / spectrally de-noised Synthetic Control.
    Rsc,
    /// Weighted average of several base methods (a model-averaging ensemble).
    /// Not a single fit — produced only by the ensemble power/evaluate paths.
    Ensemble,
}

/// The base (single-fit) methods, in canonical order. `Ensemble` is excluded:
/// it is a combination of these, not one of them.
pub const BASE_METHODS: [Method; 5] = [
    Method::Sc,
    Method::Asc,
    Method::Sdid,
    Method::Fp,
    Method::Rsc,
];

impl Method {
    pub fn name(&self) -> &'static str {
        match self {
            Method::Sc => "SC",
            Method::Asc => "ASC",
            Method::Sdid => "SDID",
            Method::Fp => "FP",
            Method::Rsc => "RSC",
            Method::Ensemble => "ENSEMBLE",
        }
    }

    /// Parse a method name (case-insensitive). `None` for anything unknown —
    /// including "ensemble", which is not a single fit.
    pub fn from_name(s: &str) -> Option<Method> {
        match s.to_ascii_lowercase().as_str() {
            "sc" => Some(Method::Sc),
            "asc" => Some(Method::Asc),
            "sdid" => Some(Method::Sdid),
            "fp" | "dsc" => Some(Method::Fp),
            "rsc" => Some(Method::Rsc),
            _ => None,
        }
    }

    /// Whether per-donor weight bounds are meaningful for this estimator.
    /// [`Method::Rsc`] regresses on de-noised factors rather than solving a
    /// simplex problem, so it has no bounded weights to constrain.
    pub fn honours_weight_bounds(&self) -> bool {
        !matches!(self, Method::Rsc)
    }
}

/// Estimator options shared by every method in a geo run.
#[derive(Clone, Copy, Debug, Default)]
pub struct FitOptions {
    /// Per-donor weight bounds (`lo ≤ w_j ≤ hi`) for the simplex-weighted
    /// methods.
    pub bounds: WeightBounds,
}

/// One point on a power curve: at a given true multiplicative lift, how often the
/// effect is detected and how the estimate is distributed.
#[derive(Clone, Debug)]
pub struct PowerPoint {
    /// True injected lift, as a fraction (0.05 = +5%).
    pub lift_pct: f64,
    /// Detection rate at this lift (the power).
    pub power: f64,
    /// Mean estimated lift (%) across simulations.
    pub est_pct_mean: f64,
    /// Percentile CI on the estimated lift (%).
    pub est_pct_lo: f64,
    pub est_pct_hi: f64,
}

/// Power-analysis result for one method.
#[derive(Clone, Debug)]
pub struct PowerResult {
    pub method: Method,
    pub points: Vec<PowerPoint>,
    /// Minimum detectable effect (smallest lift with power ≥ `target_power`),
    /// as a fraction. `None` if not reached within the grid.
    pub mde_pct: Option<f64>,
    /// MDE as an absolute per-period, per-treated-unit level change.
    pub mde_abs_per_period: Option<f64>,
    /// MDE as the cumulative incremental outcome over the whole test window
    /// (summed across treated units and post periods).
    pub mde_cumulative: Option<f64>,
    /// Critical |ATT| threshold from the historical null (level effect).
    pub crit: f64,
    /// Standard error of the estimator under the historical null (level effect).
    pub se_null: f64,
    /// Number of historical windows used as the simulation set.
    pub n_windows: usize,
}

/// Real-world diagnostics for a candidate design.
#[derive(Clone, Debug)]
pub struct Diagnostics {
    /// Treated share of total baseline volume (the "holdout"/exposure fraction).
    pub holdout_pct: f64,
    /// Pre-period fit quality: placebo RMSPE relative to the treated SD
    /// (lower is better; ~0 = near-perfect pre-fit).
    pub pre_fit_rel: f64,
    /// Improvement over a naive difference-in-differences benchmark: fraction by
    /// which the synthetic counterfactual reduces pre-period prediction error.
    pub improvement_vs_naive: f64,
    /// Strength of seasonality in the treated series (0 = none, →1 = strong),
    /// from the dominant seasonal autocorrelation.
    pub seasonality_strength: f64,
    /// Composite stability score in [0, 1] (1 = very stable pre-period).
    pub stability_score: f64,
    /// Human-readable warnings about instability / design risk.
    pub warnings: Vec<String>,
    /// Overall design confidence score in [0, 100].
    pub confidence: f64,
}
