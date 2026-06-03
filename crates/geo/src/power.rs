//! Power analysis by **historical placebo with injected lift** — the realistic
//! way to power a geo test.
//!
//! Rather than a synthetic data-generating process, we reuse the *real* panel:
//! slide a test window of length `L` across the historical series, inject a known
//! multiplicative lift on the treated units within that window, refit the
//! estimator, and record whether the effect is detected. Power at lift `τ` is the
//! detection rate across windows. The detection threshold comes from the same
//! procedure with **no** injected lift (the historical null), so it reflects the
//! data's actual noise — not an assumed variance.
//!
//! The fits are the heavy part (windows × lifts × methods × candidate markets);
//! they run in parallel via the `parallel` feature.

use crate::types::{Method, PowerPoint, PowerResult};
use panelkit_estimators::sc::{
    fit_asc_at, fit_at as fit_sc_at, fit_sdid_at, AscConfig, ScConfig, SdidConfig,
};
use panelkit_estimators::{Panel, ScFit};
use panelkit_inference::par_map_items;
use panelkit_linalg::Mat;

/// Fit the chosen estimator on a (sub-)panel with first post-period `t0`.
pub(crate) fn fit_method(panel: &Panel, t0: usize, method: Method) -> ScFit {
    match method {
        Method::Sc => fit_sc_at(panel, t0, ScConfig::default()),
        Method::Asc => fit_asc_at(panel, t0, AscConfig::default()),
        Method::Sdid => fit_sdid_at(panel, t0, SdidConfig::default()),
    }
}

/// Build the sub-panel on periods `[0, end)` with a multiplicative `lift` applied
/// to the treated units over the test window `[s, end)`.
fn injected_subpanel(y: &Mat, treated: &[usize], s: usize, end: usize, lift: f64) -> Panel {
    let n = y.rows();
    let mut m = Mat::zeros(n, end);
    for u in 0..n {
        let is_treated = treated.contains(&u);
        for t in 0..end {
            let mut v = y.get(u, t);
            if is_treated && t >= s && lift != 0.0 {
                v *= 1.0 + lift;
            }
            m.set(u, t, v);
        }
    }
    Panel::block(m, treated, s)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
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
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn std_dev(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let m = x.iter().sum::<f64>() / n as f64;
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt()
}

/// Treated-average baseline level (mean over all periods of the mean-over-treated
/// outcome) and the summed treated baseline (for the cumulative readout).
fn treated_baseline(y: &Mat, treated: &[usize]) -> (f64, f64) {
    let t = y.cols();
    let mut per_unit_mean_sum = 0.0;
    for &u in treated {
        let mut s = 0.0;
        for p in 0..t {
            s += y.get(u, p);
        }
        per_unit_mean_sum += s / t as f64;
    }
    let base_mean = per_unit_mean_sum / treated.len().max(1) as f64;
    (base_mean, per_unit_mean_sum)
}

/// Run the power analysis for one method.
///
/// `y` is the historical (untreated) panel, `treated` the candidate treatment
/// markets, `test_len` the planned test duration, `lifts` the grid of true
/// multiplicative lifts to evaluate (include `0.0` to anchor the null visually),
/// `alpha` the two-sided significance level, `target_power` for the MDE.
#[allow(clippy::too_many_arguments)]
pub fn power_curve(
    y: &Mat,
    treated: &[usize],
    test_len: usize,
    lifts: &[f64],
    method: Method,
    alpha: f64,
    target_power: f64,
    min_pre: usize,
    lookback: Option<usize>,
) -> PowerResult {
    let t = y.cols();
    assert!(test_len >= 1 && test_len < t, "test_len out of range");
    let first = min_pre.max(1);
    assert!(
        first <= t - test_len,
        "not enough periods for the requested pre-window + test_len"
    );
    // Every valid sliding test-window start position is one historical placebo.
    // We power over MANY of them (the count is `n_windows`). `lookback`, when set,
    // keeps only the most-recent K windows: those are the most representative of
    // the upcoming test (recent dynamics, longest pre-periods), at the cost of
    // fewer placebo samples.
    let mut starts: Vec<usize> = (first..=(t - test_len)).collect();
    if let Some(k) = lookback {
        let k = k.max(1);
        if starts.len() > k {
            starts = starts.split_off(starts.len() - k);
        }
    }
    let n_windows = starts.len();
    let (base_mean, base_sum) = treated_baseline(y, treated);

    // Historical null: ATT estimates with no injected lift.
    let null_atts: Vec<f64> = par_map_items(starts.clone(), |s| {
        let panel = injected_subpanel(y, treated, s, s + test_len, 0.0);
        fit_method(&panel, s, method).att
    });
    let mut abs_null: Vec<f64> = null_atts.iter().map(|a| a.abs()).collect();
    abs_null.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let crit = quantile(&abs_null, 1.0 - alpha);
    let se_null = std_dev(&null_atts);

    // Power at each lift.
    let mut points = Vec::with_capacity(lifts.len());
    for &lift in lifts {
        let atts: Vec<f64> = if lift == 0.0 {
            null_atts.clone()
        } else {
            par_map_items(starts.clone(), |s| {
                let panel = injected_subpanel(y, treated, s, s + test_len, lift);
                fit_method(&panel, s, method).att
            })
        };
        let power = atts.iter().filter(|a| a.abs() > crit).count() as f64 / n_windows as f64;
        // Convert level estimates to % of treated baseline.
        let mut est_pct: Vec<f64> = atts.iter().map(|a| a / base_mean).collect();
        let mean_pct = est_pct.iter().sum::<f64>() / est_pct.len() as f64;
        est_pct.sort_by(|a, b| a.partial_cmp(b).unwrap());
        points.push(PowerPoint {
            lift_pct: lift,
            power,
            est_pct_mean: mean_pct,
            est_pct_lo: quantile(&est_pct, alpha / 2.0),
            est_pct_hi: quantile(&est_pct, 1.0 - alpha / 2.0),
        });
    }

    // MDE: smallest lift reaching target_power (linear interpolation on the grid).
    let mde_pct = mde_from_points(&points, target_power);
    let (mde_abs_per_period, mde_cumulative) = match mde_pct {
        Some(m) => (Some(m * base_mean), Some(m * base_sum * test_len as f64)),
        None => (None, None),
    };

    PowerResult {
        method,
        points,
        mde_pct,
        mde_abs_per_period,
        mde_cumulative,
        crit,
        se_null,
        n_windows,
    }
}

/// Smallest lift with power ≥ `target`, interpolating between bracketing grid
/// points. Assumes `points` are in ascending lift order.
fn mde_from_points(points: &[PowerPoint], target: f64) -> Option<f64> {
    let mut prev: Option<&PowerPoint> = None;
    for p in points {
        if p.power >= target {
            return Some(match prev {
                Some(q) if p.power > q.power => {
                    // linear interpolation in (power, lift)
                    let frac = (target - q.power) / (p.power - q.power);
                    q.lift_pct + frac * (p.lift_pct - q.lift_pct)
                }
                _ => p.lift_pct,
            });
        }
        prev = Some(p);
    }
    None
}
