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
/// estimates: `sqrt((n-1)/n · Σ(θ_i − θ̄)²)`, centered on the LOO mean.
pub fn jackknife_se(loo_estimates: &[f64]) -> f64 {
    let n = loo_estimates.len();
    if n < 2 {
        return 0.0;
    }
    let mean = loo_estimates.iter().sum::<f64>() / n as f64;
    let ss: f64 = loo_estimates.iter().map(|x| (x - mean).powi(2)).sum();
    ((n as f64 - 1.0) / n as f64 * ss).sqrt()
}

#[inline]
fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

/// Moving-block bootstrap of the **mean** of a (serially-dependent) series.
///
/// Resamples length-`block_len` contiguous (circular) blocks until a series of
/// the original length is filled, then takes its mean; repeats `n_reps` times.
/// Accounts for serial correlation in, e.g., a synthetic-control gap path when
/// estimating the sampling variability of the average post-period effect.
///
/// Determinism: replicate `b` draws from substream `(seed, b)`.
pub fn block_bootstrap_mean(
    series: &[f64],
    block_len: usize,
    n_reps: usize,
    seed: u64,
    level: f64,
) -> (ConfidenceInterval, Vec<f64>) {
    let t = series.len();
    let point = mean(series);
    if t == 0 {
        return (percentile_ci(point, &[point], level), Vec::new());
    }
    let l = block_len.clamp(1, t);
    let draws = par_map(n_reps, seed, |_, rng| {
        let mut acc = 0.0;
        let mut count = 0usize;
        while count < t {
            let start = rng.gen_range(t);
            for k in 0..l {
                if count >= t {
                    break;
                }
                acc += series[(start + k) % t];
                count += 1;
            }
        }
        acc / t as f64
    });
    (percentile_ci(point, &draws, level), draws)
}

/// Politis–Romano **stationary** bootstrap of the mean of a series.
///
/// Like the moving-block bootstrap but with random geometric block lengths
/// (mean `mean_block_len`): at each step, continue to the next (circular) index
/// with probability `1 − 1/L`, else jump to a fresh random index. Produces a
/// strictly stationary resampled series.
pub fn stationary_bootstrap_mean(
    series: &[f64],
    mean_block_len: usize,
    n_reps: usize,
    seed: u64,
    level: f64,
) -> (ConfidenceInterval, Vec<f64>) {
    let t = series.len();
    let point = mean(series);
    if t == 0 {
        return (percentile_ci(point, &[point], level), Vec::new());
    }
    let p = 1.0 / mean_block_len.max(1) as f64;
    let draws = par_map(n_reps, seed, |_, rng| {
        let mut idx = rng.gen_range(t);
        let mut acc = 0.0;
        for _ in 0..t {
            acc += series[idx];
            if rng.next_f64() < p {
                idx = rng.gen_range(t);
            } else {
                idx = (idx + 1) % t;
            }
        }
        acc / t as f64
    });
    (percentile_ci(point, &draws, level), draws)
}
