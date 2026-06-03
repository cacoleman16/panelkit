//! Real-world design diagnostics: is this geo design trustworthy *before* you
//! spend money on it?
//!
//! Covers holdout share, pre-period fit quality, improvement over a naive DiD
//! benchmark, seasonality strength, a pre-period stability score, plain-language
//! warnings, and a composite 0–100 confidence score.

use crate::power::fit_method;
use crate::types::{Diagnostics, Method};
use panelkit_estimators::Panel;
use panelkit_linalg::Mat;

/// Mean over a set of units at each period → length-T series.
fn unit_avg(y: &Mat, units: &[usize]) -> Vec<f64> {
    let t = y.cols();
    let mut out = vec![0.0; t];
    for &u in units {
        for p in 0..t {
            out[p] += y.get(u, p);
        }
    }
    let inv = 1.0 / units.len().max(1) as f64;
    out.iter_mut().for_each(|v| *v *= inv);
    out
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len().max(1) as f64
}

fn std_dev(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt()
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let s: f64 = (0..n).map(|i| (a[i] - b[i]).powi(2)).sum();
    (s / n as f64).sqrt()
}

/// Strength of seasonality: the largest autocorrelation of the
/// first-differenced series at lags ≥ 2 (in [0, 1)). First-differencing removes
/// the trend so the ACF reflects periodicity, not drift.
fn seasonality_strength(series: &[f64]) -> (f64, usize) {
    let t = series.len();
    if t < 6 {
        return (0.0, 0);
    }
    let d: Vec<f64> = (1..t).map(|i| series[i] - series[i - 1]).collect();
    let m = mean(&d);
    let denom: f64 = d.iter().map(|v| (v - m).powi(2)).sum();
    if denom <= 0.0 {
        return (0.0, 0);
    }
    let max_lag = (d.len() / 2).min(60);
    let mut best = 0.0_f64;
    let mut best_lag = 0;
    for lag in 2..=max_lag {
        let mut num = 0.0;
        for i in lag..d.len() {
            num += (d[i] - m) * (d[i - lag] - m);
        }
        let ac = num / denom;
        if ac > best {
            best = ac;
            best_lag = lag;
        }
    }
    (best.clamp(0.0, 1.0), best_lag)
}

/// Compute design diagnostics for `treated` on the historical panel `y`, using
/// the planned window = the last `test_len` periods.
pub fn diagnostics(y: &Mat, treated: &[usize], test_len: usize) -> Diagnostics {
    let (n, t) = y.shape();
    let t0 = t - test_len;
    let controls: Vec<usize> = (0..n).filter(|u| !treated.contains(u)).collect();

    let treated_series = unit_avg(y, treated);
    let control_series = unit_avg(y, &controls);

    // --- Holdout share (treated baseline / total baseline). ---
    let total_base: f64 = (0..n)
        .map(|u| (0..t).map(|p| y.get(u, p)).sum::<f64>())
        .sum();
    let treated_base: f64 = treated
        .iter()
        .map(|&u| (0..t).map(|p| y.get(u, p)).sum::<f64>())
        .sum();
    let holdout_pct = if total_base > 0.0 {
        treated_base / total_base
    } else {
        0.0
    };

    // --- Pre-period fit quality from a real SC fit on the planned window. ---
    let panel = Panel::block(y.clone(), treated, t0);
    let fit = fit_method(&panel, t0, Method::Sc);
    let treated_pre = &treated_series[..t0];
    let pre_sd = std_dev(treated_pre).max(1e-12);
    let pre_fit_rel = fit.pre_rmspe / pre_sd;

    // --- Improvement over a naive DiD benchmark (pre-period prediction). ---
    // Naive counterfactual for the treated avg = control avg shifted to match the
    // pre-period level. Compare its pre-period error to SC's.
    let shift = mean(treated_pre) - mean(&control_series[..t0]);
    let naive_pre: Vec<f64> = control_series[..t0].iter().map(|c| c + shift).collect();
    let naive_rmse = rmse(treated_pre, &naive_pre).max(1e-12);
    let improvement_vs_naive = (1.0 - fit.pre_rmspe / naive_rmse).clamp(0.0, 1.0);

    // --- Seasonality + stability. ---
    let (seasonality_strength_val, season_lag) = seasonality_strength(treated_pre);
    // Volatility = SD of first differences relative to the level.
    let diffs: Vec<f64> = (1..t0)
        .map(|i| treated_pre[i] - treated_pre[i - 1])
        .collect();
    let level = mean(treated_pre).abs().max(1e-12);
    let volatility = std_dev(&diffs) / level;
    let stability_score = (1.0 / (1.0 + 8.0 * volatility)).clamp(0.0, 1.0);

    // --- Warnings. ---
    let mut warnings = Vec::new();
    if pre_fit_rel > 0.5 {
        warnings.push(format!(
            "Weak pre-period fit (relative RMSPE {:.2}); the synthetic control tracks the treated markets poorly — treat results with caution.",
            pre_fit_rel
        ));
    }
    if volatility > 0.25 {
        warnings.push(format!(
            "Treated markets are volatile pre-period (period-to-period swings ≈{:.0}% of level); power may be unstable.",
            100.0 * volatility
        ));
    }
    if seasonality_strength_val > 0.3 {
        let cycles = if season_lag > 0 {
            t0 as f64 / season_lag as f64
        } else {
            0.0
        };
        if cycles < 2.0 {
            warnings.push(format!(
                "Strong seasonality (≈{}-period cycle) but the pre-window covers only {:.1} cycles; use ≥2 full cycles of history.",
                season_lag, cycles
            ));
        } else {
            warnings.push(format!(
                "Seasonality detected (≈{}-period cycle); ensure the test window spans comparable seasonal conditions.",
                season_lag
            ));
        }
    }
    if holdout_pct < 0.02 {
        warnings.push(format!(
            "Treated markets are only {:.1}% of volume — likely too small to detect realistic lifts.",
            100.0 * holdout_pct
        ));
    } else if holdout_pct > 0.45 {
        warnings.push(format!(
            "Treated markets are {:.0}% of volume — a large holdout leaves a thin, possibly poor-matching donor pool.",
            100.0 * holdout_pct
        ));
    }
    if controls.len() < 5 {
        warnings.push(format!(
            "Only {} donor markets; synthetic control needs a richer donor pool for a stable counterfactual.",
            controls.len()
        ));
    }
    if t0 < 2 * test_len {
        warnings.push(format!(
            "Short pre-period ({} periods) relative to the {}-period test; more history improves the fit.",
            t0, test_len
        ));
    }

    // --- Composite confidence (0–100). ---
    let fit_component = (1.0 - pre_fit_rel).clamp(0.0, 1.0);
    let holdout_adequacy = {
        // Best in roughly [3%, 35%]; penalize extremes.
        let h = holdout_pct;
        if (0.03..=0.35).contains(&h) {
            1.0
        } else if h < 0.03 {
            (h / 0.03).clamp(0.0, 1.0)
        } else {
            (1.0 - (h - 0.35) / 0.35).clamp(0.0, 1.0)
        }
    };
    let confidence = 100.0
        * (0.40 * fit_component
            + 0.25 * stability_score
            + 0.15 * holdout_adequacy
            + 0.20 * improvement_vs_naive)
            .clamp(0.0, 1.0);

    Diagnostics {
        holdout_pct,
        pre_fit_rel,
        improvement_vs_naive,
        seasonality_strength: seasonality_strength_val,
        stability_score,
        warnings,
        confidence,
    }
}
