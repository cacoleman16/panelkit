//! Multiplier (wild) bootstrap on influence functions.
//!
//! Given a statistic's unit-level influence functions `IF_i` (e.g. from
//! Callaway–Sant'Anna), the multiplier bootstrap draws mean-zero, unit-variance
//! weights `V_i` per replicate and forms `θ̂* = (1/N) Σ_i V_i · IF_i`. The
//! standard deviation of these draws estimates the SE; their quantiles give a
//! confidence interval. This is the Callaway–Sant'Anna-recommended inference
//! method and is the natural way to get uniform event-study bands.
//!
//! Determinism: replicate `b` uses substream `(seed, b)`, so results are
//! independent of thread count.

use crate::ci::{percentile_ci, ConfidenceInterval};
use crate::parallel::par_map;

/// Mammen two-point weights (mean 0, variance 1, skewness 1) — the standard
/// wild-bootstrap multiplier.
#[inline]
fn mammen(rng: &mut panelkit_linalg::rng::Xoshiro256pp) -> f64 {
    const P: f64 = 0.723606797749979; // (√5 + 1) / (2√5)
    const A: f64 = -0.618033988749895; // -(√5 − 1)/2
    const B: f64 = 1.618033988749895; //  (√5 + 1)/2
    if rng.next_f64() < P {
        A
    } else {
        B
    }
}

/// Multiplier-bootstrap a statistic from its influence functions.
///
/// Returns the resampling distribution and a percentile confidence interval
/// centered at `point`.
pub fn multiplier_bootstrap(
    influence: &[f64],
    point: f64,
    n_reps: usize,
    seed: u64,
    level: f64,
) -> (ConfidenceInterval, Vec<f64>) {
    let n = influence.len();
    let draws = par_map(n_reps, seed, |_, rng| {
        let mut acc = 0.0;
        for &ifi in influence {
            acc += mammen(rng) * ifi;
        }
        acc / n as f64
    });
    // Center the bootstrap deviations around the point estimate.
    let centered: Vec<f64> = draws.iter().map(|d| point + d).collect();
    let ci = percentile_ci(point, &centered, level);
    (ci, centered)
}

/// Jackknife (leave-one-out) standard error from a set of leave-one-out
/// estimates and the full-sample estimate.
pub fn jackknife_se(loo_estimates: &[f64], full: f64) -> f64 {
    let n = loo_estimates.len();
    if n < 2 {
        return 0.0;
    }
    let mean = loo_estimates.iter().sum::<f64>() / n as f64;
    let _ = full;
    let ss: f64 = loo_estimates.iter().map(|x| (x - mean).powi(2)).sum();
    ((n as f64 - 1.0) / n as f64 * ss).sqrt()
}
