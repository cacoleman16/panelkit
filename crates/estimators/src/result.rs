//! Shared result types returned by estimators.

/// The result of a synthetic-control-style fit (SC, ASC, SDID, MC-NNM all
/// produce a counterfactual path and an ATT).
#[derive(Clone, Debug)]
pub struct ScFit {
    /// Donor weights (length = number of donors). Empty for methods that do not
    /// produce explicit unit weights (e.g. MC-NNM).
    pub weights: Vec<f64>,
    /// Donor unit indices matching `weights`.
    pub donor_ids: Vec<usize>,
    /// Per-post-period treatment effect: treated − counterfactual.
    pub att_path: Vec<f64>,
    /// Average post-period ATT.
    pub att: f64,
    /// Estimated counterfactual on the post-period (length = T_post).
    pub counterfactual_post: Vec<f64>,
    /// Observed (aggregated) treated outcome on the post-period.
    pub treated_post: Vec<f64>,
    /// Pre-period root-mean-squared prediction error (fit quality).
    pub pre_rmspe: f64,
    /// Post-period root-mean-squared prediction error.
    pub post_rmspe: f64,
}

impl ScFit {
    /// Ratio of post- to pre-period RMSPE — the Abadie placebo test statistic.
    pub fn rmspe_ratio(&self) -> f64 {
        if self.pre_rmspe > 0.0 {
            self.post_rmspe / self.pre_rmspe
        } else {
            f64::INFINITY
        }
    }
}

/// The result of a difference-in-differences fit producing a scalar ATT plus an
/// optional event-study path.
#[derive(Clone, Debug)]
pub struct DidFit {
    /// Overall ATT.
    pub att: f64,
    /// Analytic standard error, when available from the estimator.
    pub se: Option<f64>,
    /// Event-study relative periods (e.g. -3, -2, -1, 0, 1, ...), if estimated.
    pub event_time: Vec<i64>,
    /// Event-study coefficients matching `event_time`.
    pub event_coef: Vec<f64>,
    /// Event-study standard errors matching `event_time`, if available.
    pub event_se: Vec<f64>,
}
