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

/// Inverse standard-normal CDF (Acklam's rational approximation, |ε| < 1.2e-9
/// over (0, 1)). Used for normal-approximation intervals (e.g. jackknife).
pub fn normal_quantile(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "normal_quantile needs p in (0, 1)");
    // Coefficients from Peter Acklam's algorithm (public domain).
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}
