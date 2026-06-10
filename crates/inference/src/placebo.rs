//! Placebo / permutation inference for the synthetic-control family
//! (Abadie et al. 2010).
//!
//! In-space placebo: reassign treatment, in turn, to each never-treated donor
//! (using the remaining donors as its pool), refit **the same estimator** the
//! point estimate used (SC, ASC, or SDID), and record the post/pre RMSPE
//! ratio. The treated unit's ratio is then compared against this placebo
//! distribution; the one-sided p-value is the share of placebo ratios at least
//! as extreme. The placebo ATTs (outcome units) double as the null
//! distribution for an ATT-scale SE / CI.

use panelkit_estimators::sc::augmented::AscConfig;
use panelkit_estimators::sc::sdid::SdidConfig;
use panelkit_estimators::sc::synthetic::ScConfig;
use panelkit_estimators::sc::{fit_asc_at, fit_at as fit_sc_at, fit_sdid_at};
use panelkit_estimators::{Panel, ScFit};
use panelkit_linalg::Mat;

/// Which SC-family estimator the placebo engine refits per donor.
#[derive(Clone, Copy, Debug)]
pub enum ScMethod {
    Sc(ScConfig),
    Asc(AscConfig),
    Sdid(SdidConfig),
}

impl ScMethod {
    fn fit_at(&self, panel: &Panel, t0: usize) -> ScFit {
        match *self {
            ScMethod::Sc(cfg) => fit_sc_at(panel, t0, cfg),
            ScMethod::Asc(cfg) => fit_asc_at(panel, t0, cfg),
            ScMethod::Sdid(cfg) => fit_sdid_at(panel, t0, cfg),
        }
    }

    /// The placebo test statistic for a fit. SC/ASC use Abadie's post/pre
    /// RMSPE ratio. SDID uses |ATT|: its `post_rmspe` measures dispersion of
    /// the gap path *around its mean*, so a constant treatment effect leaves
    /// the ratio completely unchanged — the ratio statistic has no power
    /// against exactly the alternative being tested.
    fn statistic(&self, fit: &ScFit) -> f64 {
        match self {
            ScMethod::Sdid(_) => fit.att.abs(),
            _ => fit.rmspe_ratio(),
        }
    }
}

/// Outcome of an SC-family placebo test.
#[derive(Clone, Debug)]
pub struct PlaceboResult {
    /// Estimated ATT for the real treated unit(s).
    pub att: f64,
    /// Per-post-period ATT path.
    pub att_path: Vec<f64>,
    /// Treated unit's test statistic (post/pre RMSPE ratio for SC/ASC;
    /// |ATT| for SDID — see [`ScMethod`]).
    pub treated_ratio: f64,
    /// Placebo test statistics, one per donor — these drive the p-value.
    pub placebo_ratios: Vec<f64>,
    /// Placebo ATTs (mean post-period gap per placebo fit), in **outcome
    /// units**, aligned with `placebo_ratios`. Under no effect these are null
    /// draws of the ATT estimator — the right reference for an ATT-scale SE
    /// and confidence interval. (The ratios are not: they are unitless.)
    pub placebo_atts: Vec<f64>,
    /// One-sided p-value: P(placebo ratio ≥ treated ratio).
    pub p_value: f64,
}

/// Build the donor-only sub-panel where donor `d` plays "treated" (row 0) and
/// the remaining donors are its pool. Excluding the actual treated unit(s)
/// keeps the real effect from leaking into the placebo null.
fn placebo_panel(panel: &Panel, donors: &[usize], idx: usize, t0: usize) -> Panel {
    let t = panel.n_periods();
    let mut rows: Vec<usize> = Vec::with_capacity(donors.len());
    rows.push(donors[idx]);
    rows.extend(
        donors
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != idx)
            .map(|(_, &u)| u),
    );
    let mut y = Mat::zeros(rows.len(), t);
    for (r, &u) in rows.iter().enumerate() {
        for p in 0..t {
            y.set(r, p, panel.outcome(u, p));
        }
    }
    Panel::block(y, &[0], t0)
}

/// Run the in-space placebo test, refitting `method` per donor.
pub fn sc_placebo_method(panel: &Panel, method: ScMethod) -> PlaceboResult {
    let t0 = panel
        .common_treat_time()
        .expect("placebo test requires a single common treatment time");

    // Real treated fit.
    let treated_fit = method.fit_at(panel, t0);
    let treated_ratio = method.statistic(&treated_fit);

    let donors = panel.never_treated_units();

    // Each donor's placebo fit is independent of the others, so we farm them out
    // in parallel (when the `parallel` feature is on); the per-donor result does
    // not depend on ordering, so the output is deterministic regardless.
    let work: Vec<usize> = (0..donors.len()).collect();
    let placebo_fits: Vec<(f64, f64)> = crate::parallel::par_map_items(work, |idx| {
        if donors.len() < 2 {
            return (f64::NAN, f64::NAN);
        }
        let sub = placebo_panel(panel, &donors, idx, t0);
        let fit = method.fit_at(&sub, t0);
        (method.statistic(&fit), fit.att)
    })
    .into_iter()
    .filter(|(r, _)| !r.is_nan())
    .collect();
    let placebo_ratios: Vec<f64> = placebo_fits.iter().map(|&(r, _)| r).collect();
    let placebo_atts: Vec<f64> = placebo_fits.iter().map(|&(_, a)| a).collect();

    let n = placebo_ratios.len();
    let n_extreme = placebo_ratios
        .iter()
        .filter(|&&r| r >= treated_ratio)
        .count();
    // +1 smoothing (include the treated unit itself) for a valid finite-sample
    // permutation p-value.
    let p_value = (1 + n_extreme) as f64 / (1 + n) as f64;

    PlaceboResult {
        att: treated_fit.att,
        att_path: treated_fit.att_path,
        treated_ratio,
        placebo_ratios,
        placebo_atts,
        p_value,
    }
}

/// Run the SC in-space placebo test (back-compat wrapper).
pub fn sc_placebo(panel: &Panel, cfg: ScConfig) -> PlaceboResult {
    sc_placebo_method(panel, ScMethod::Sc(cfg))
}
