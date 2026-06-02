//! Confidence-interval construction from resampling distributions.

/// A point estimate with an associated interval and standard error.
#[derive(Clone, Copy, Debug)]
pub struct ConfidenceInterval {
    pub point: f64,
    pub se: f64,
    pub lower: f64,
    pub upper: f64,
    pub level: f64,
}

/// Percentile confidence interval from a vector of resampled statistics.
/// `level` is the coverage (e.g. 0.95).
pub fn percentile_ci(point: f64, draws: &[f64], level: f64) -> ConfidenceInterval {
    assert!(level > 0.0 && level < 1.0);
    let mut sorted = draws.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let alpha = 1.0 - level;
    let lo = quantile_sorted(&sorted, alpha / 2.0);
    let hi = quantile_sorted(&sorted, 1.0 - alpha / 2.0);
    // Standard error = sample SD of the draws.
    let mean = draws.iter().sum::<f64>() / n.max(1) as f64;
    let var = if n > 1 {
        draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
    } else {
        0.0
    };
    ConfidenceInterval {
        point,
        se: var.sqrt(),
        lower: lo,
        upper: hi,
        level,
    }
}

/// Linearly-interpolated quantile of an already-sorted slice.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}
