//! Placebo / permutation inference for synthetic control (Abadie et al. 2010).
//!
//! In-space placebo: reassign treatment, in turn, to each never-treated donor
//! (using the remaining donors as its pool), refit, and record the post/pre
//! RMSPE ratio. The treated unit's ratio is then compared against this placebo
//! distribution; the one-sided p-value is the share of placebo ratios at least
//! as extreme.

use panelkit_estimators::sc::synthetic::{fit_at, fit_series, ScConfig};
use panelkit_estimators::Panel;
use panelkit_linalg::Mat;

/// Outcome of an SC placebo test.
#[derive(Clone, Debug)]
pub struct PlaceboResult {
    /// Estimated ATT for the real treated unit(s).
    pub att: f64,
    /// Per-post-period ATT path.
    pub att_path: Vec<f64>,
    /// Treated unit's post/pre RMSPE ratio (the test statistic).
    pub treated_ratio: f64,
    /// Placebo RMSPE ratios, one per donor.
    pub placebo_ratios: Vec<f64>,
    /// One-sided p-value: P(placebo ratio ≥ treated ratio).
    pub p_value: f64,
}

/// Build the pre/post donor blocks for an explicit list of donor units.
fn donor_blocks(panel: &Panel, donors: &[usize], t0: usize) -> (Mat, Mat) {
    let t = panel.n_periods();
    let mut pre = Mat::zeros(t0, donors.len());
    let mut post = Mat::zeros(t - t0, donors.len());
    for (jc, &u) in donors.iter().enumerate() {
        for p in 0..t0 {
            pre.set(p, jc, panel.outcome(u, p));
        }
        for p in t0..t {
            post.set(p - t0, jc, panel.outcome(u, p));
        }
    }
    (pre, post)
}

/// Run the SC in-space placebo test.
pub fn sc_placebo(panel: &Panel, cfg: ScConfig) -> PlaceboResult {
    let t0 = panel
        .common_treat_time()
        .expect("placebo test requires a single common treatment time");

    // Real treated fit.
    let treated_fit = fit_at(panel, t0, cfg);
    let treated_ratio = treated_fit.rmspe_ratio();

    let donors = panel.never_treated_units();
    let t = panel.n_periods();

    // Each donor's placebo fit is independent of the others, so we farm them out
    // in parallel (when the `parallel` feature is on); the per-donor result does
    // not depend on ordering, so the output is deterministic regardless.
    let work: Vec<usize> = (0..donors.len()).collect();
    let placebo_ratios: Vec<f64> = crate::parallel::par_map_items(work, |idx| {
        let d = donors[idx];
        let pool: Vec<usize> = donors
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != idx)
            .map(|(_, &u)| u)
            .collect();
        if pool.is_empty() {
            return f64::NAN;
        }
        let (pre, post) = donor_blocks(panel, &pool, t0);
        let y_pre: Vec<f64> = (0..t0).map(|p| panel.outcome(d, p)).collect();
        let y_post: Vec<f64> = (t0..t).map(|p| panel.outcome(d, p)).collect();
        let fit = fit_series(&y_pre, &y_post, &pre, &post, pool, cfg.ridge);
        fit.rmspe_ratio()
    })
    .into_iter()
    .filter(|r| !r.is_nan())
    .collect();

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
        p_value,
    }
}
