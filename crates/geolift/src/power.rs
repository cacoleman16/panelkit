//! Power analysis & minimum detectable effect (MDE) for geo test-market design.
//!
//! The question a geo test asks before spending money: *if I run the treatment
//! on these markets for this many periods, how big a lift could I actually
//! detect?* We answer it the way it's done in practice — by replaying the real
//! historical data many times as pseudo-experiments:
//!
//! 1. Slide a treatment window of length `eval_window` to many placements in the
//!    historical period (this captures the real week-to-week variability — no
//!    synthetic noise assumed).
//! 2. At each placement, designate the test markets as "treated", multiply their
//!    outcomes by `(1 + lift)` over the window, and fit the estimator on the
//!    pre-window data.
//! 3. The `lift = 0` runs give the **null distribution** of the estimate; its
//!    `1 − α` quantile is the rejection threshold. For each candidate `lift`,
//!    **power** is the share of runs whose estimate clears that threshold.
//! 4. The **MDE** is the smallest lift reaching the target power (e.g. 0.8).
//!
//! This is a time-placement permutation calibration — honest about the data's
//! own noise, and identical in spirit to how practitioners size geo tests.

use panelkit_estimators::sc::{fit_asc_at, fit_at as sc_fit_at, fit_sdid_at, AscConfig, ScConfig, SdidConfig};
use panelkit_estimators::Panel;
use panelkit_linalg::rng::Xoshiro256pp;
use panelkit_linalg::Mat;

/// Estimator used for the power study.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Synthetic Control.
    Sc,
    /// Augmented SC.
    Asc,
    /// Synthetic DiD.
    Sdid,
    /// Naive 2×2 difference-in-differences (no weights) — the baseline to beat.
    Naive,
}

impl Method {
    pub fn label(&self) -> &'static str {
        match self {
            Method::Sc => "SC",
            Method::Asc => "ASC",
            Method::Sdid => "SDID",
            Method::Naive => "Naive DiD",
        }
    }
}

/// Configuration for a power study.
#[derive(Clone, Debug)]
pub struct PowerConfig {
    /// Relative lifts to evaluate (e.g. `[0.0, 0.01, 0.02, 0.05, 0.10]`). `0.0`
    /// is required (it calibrates the null threshold).
    pub effects: Vec<f64>,
    /// Treatment duration in periods.
    pub eval_window: usize,
    /// Minimum pre-window periods each pseudo-experiment must keep.
    pub min_pre: usize,
    /// Number of pseudo-experiments per effect size.
    pub n_sims: usize,
    /// Two-sided significance level.
    pub alpha: f64,
    /// Target power for the MDE (e.g. 0.8).
    pub power_target: f64,
    pub seed: u64,
}

impl Default for PowerConfig {
    fn default() -> Self {
        PowerConfig {
            effects: vec![0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20],
            eval_window: 14,
            min_pre: 0, // 0 => auto (half the history)
            n_sims: 250,
            alpha: 0.10,
            power_target: 0.8,
            seed: 0,
        }
    }
}

/// Power-curve result for one method.
#[derive(Clone, Debug)]
pub struct PowerResult {
    pub method: Method,
    pub effects: Vec<f64>,
    pub power: Vec<f64>,
    /// Smallest lift reaching `power_target` (linearly interpolated), if any.
    pub mde: Option<f64>,
    /// Null (1−α) rejection threshold on |ATT estimate|.
    pub null_threshold: f64,
    /// Number of distinct treatment-window placements available.
    pub distinct_windows: usize,
}

/// Run a power study for one estimator on a given test-market set.
///
/// `y` is the historical `N×T` panel; `test_markets` are the row indices to be
/// "treated" in the pseudo-experiments; all other rows are donors/controls.
pub fn power_curve(y: &Mat, test_markets: &[usize], method: Method, cfg: &PowerConfig) -> PowerResult {
    let (n, t) = y.shape();
    let l = cfg.eval_window.max(1);
    let min_pre = if cfg.min_pre == 0 { (t / 2).max(2) } else { cfg.min_pre };
    // Valid pseudo-treatment starts: need min_pre pre-periods and L post-periods.
    let lo = min_pre;
    let hi = t.saturating_sub(l); // inclusive upper bound for t0
    let distinct = if hi >= lo { hi - lo + 1 } else { 0 };
    assert!(distinct > 0, "history too short for eval_window + min_pre");

    let is_test = {
        let mut v = vec![false; n];
        for &m in test_markets {
            assert!(m < n, "test market index out of range");
            v[m] = true;
        }
        v
    };
    let donors: Vec<usize> = (0..n).filter(|&i| !is_test[i]).collect();
    assert!(!donors.is_empty(), "need at least one donor market");

    // One pseudo-experiment: place window at t0, inject lift, fit, return ATT.
    let run = |t0: usize, lift: f64| -> f64 {
        let end = t0 + l;
        // Sub-panel over [0, end): test markets treated at t0.
        let mut sub = Mat::zeros(n, end);
        for j in 0..end {
            for i in 0..n {
                let mut v = y.get(i, j);
                if is_test[i] && j >= t0 {
                    v *= 1.0 + lift;
                }
                sub.set(i, j, v);
            }
        }
        let panel = Panel::block(sub, test_markets, t0);
        match method {
            Method::Sc => sc_fit_at(&panel, t0, ScConfig::default()).att,
            Method::Asc => fit_asc_at(&panel, t0, AscConfig::default()).att,
            Method::Sdid => fit_sdid_at(&panel, t0, SdidConfig::default()).att,
            Method::Naive => naive_did(y, &is_test, &donors, t0, l, lift),
        }
    };

    // Estimates per effect, across pseudo-experiments (deterministic substreams).
    let estimates: Vec<Vec<f64>> = cfg
        .effects
        .iter()
        .enumerate()
        .map(|(ei, &lift)| {
            run_sims(cfg.n_sims, cfg.seed.wrapping_add(ei as u64 * 7919), lo, hi, |t0| {
                run(t0, lift)
            })
        })
        .collect();

    // Null threshold from the lift==0 effect (or the smallest effect present).
    let null_idx = cfg
        .effects
        .iter()
        .position(|&e| e == 0.0)
        .unwrap_or(0);
    let mut null_abs: Vec<f64> = estimates[null_idx].iter().map(|x| x.abs()).collect();
    null_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold = quantile_sorted(&null_abs, 1.0 - cfg.alpha);

    // Power per effect.
    let power: Vec<f64> = estimates
        .iter()
        .map(|est| {
            let hits = est.iter().filter(|&&e| e.abs() > threshold).count();
            hits as f64 / est.len().max(1) as f64
        })
        .collect();

    let mde = interpolate_mde(&cfg.effects, &power, cfg.power_target);

    PowerResult {
        method,
        effects: cfg.effects.clone(),
        power,
        mde,
        null_threshold: threshold,
        distinct_windows: distinct,
    }
}

/// Naive 2×2 DiD: (test post−pre change) − (donor post−pre change), on the
/// aggregated means, with the same multiplicative lift injected on test markets.
fn naive_did(y: &Mat, is_test: &[bool], donors: &[usize], t0: usize, l: usize, lift: f64) -> f64 {
    let test: Vec<usize> = (0..is_test.len()).filter(|&i| is_test[i]).collect();
    let mean = |units: &[usize], lo: usize, hi: usize, inject: bool| -> f64 {
        let mut s = 0.0;
        for &u in units {
            for j in lo..hi {
                let mut v = y.get(u, j);
                if inject {
                    v *= 1.0 + lift;
                }
                s += v;
            }
        }
        s / (units.len() * (hi - lo)) as f64
    };
    let test_pre = mean(&test, 0, t0, false);
    let test_post = mean(&test, t0, t0 + l, true);
    let don_pre = mean(donors, 0, t0, false);
    let don_post = mean(donors, t0, t0 + l, false);
    (test_post - test_pre) - (don_post - don_pre)
}

/// Run `n` pseudo-experiments drawing `t0` uniformly from `[lo, hi]` with a
/// deterministic, thread-invariant substream per replicate.
#[cfg(feature = "parallel")]
fn run_sims<F>(n: usize, seed: u64, lo: usize, hi: usize, f: F) -> Vec<f64>
where
    F: Fn(usize) -> f64 + Sync + Send,
{
    use rayon::prelude::*;
    (0..n)
        .into_par_iter()
        .map(|s| {
            let mut rng = Xoshiro256pp::substream(seed, s as u64);
            let t0 = lo + rng.gen_range(hi - lo + 1);
            f(t0)
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn run_sims<F>(n: usize, seed: u64, lo: usize, hi: usize, f: F) -> Vec<f64>
where
    F: Fn(usize) -> f64,
{
    (0..n)
        .map(|s| {
            let mut rng = Xoshiro256pp::substream(seed, s as u64);
            let t0 = lo + rng.gen_range(hi - lo + 1);
            f(t0)
        })
        .collect()
}

/// Linearly-interpolated quantile of a sorted slice.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::INFINITY;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n as f64 - 1.0);
    let l = pos.floor() as usize;
    let h = pos.ceil() as usize;
    if l == h {
        sorted[l]
    } else {
        let frac = pos - l as f64;
        sorted[l] * (1.0 - frac) + sorted[h] * frac
    }
}

/// Smallest effect reaching `target` power, linearly interpolating between the
/// two grid points that bracket the crossing. `None` if never reached.
fn interpolate_mde(effects: &[f64], power: &[f64], target: f64) -> Option<f64> {
    for i in 0..effects.len() {
        if power[i] >= target {
            if i == 0 {
                return Some(effects[i]);
            }
            let (e0, e1) = (effects[i - 1], effects[i]);
            let (p0, p1) = (power[i - 1], power[i]);
            if p1 > p0 {
                let frac = (target - p0) / (p1 - p0);
                return Some(e0 + frac * (e1 - e0));
            }
            return Some(effects[i]);
        }
    }
    None
}
