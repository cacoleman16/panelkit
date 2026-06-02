//! Synthetic-control estimator correctness on planted-effect panels.

use panelkit_estimators::sc::{fit_sc, ScConfig};
use panelkit_estimators::Panel;
use panelkit_linalg::rng::Xoshiro256pp;
use panelkit_linalg::Mat;

/// Build a panel where the treated unit is, in the pre-period, an exact convex
/// combination of two donors; in the post-period we add a known constant effect
/// `tau`. SC should recover the counterfactual and hence `tau`.
fn planted_panel(tau: f64, seed: u64) -> (Panel, usize) {
    let n = 6usize; // units: 0 = treated, 1..6 donors
    let t = 20usize;
    let t0 = 14usize;
    let mut rng = Xoshiro256pp::seed_from_u64(seed);

    let mut y = Mat::zeros(n, t);
    // Donors get smooth random walks.
    for u in 1..n {
        let mut level = rng.next_normal();
        for p in 0..t {
            level += 0.3 * rng.next_normal();
            y.set(u, p, 10.0 + level + 0.5 * (u as f64));
        }
    }
    // Treated (unit 0) = 0.6*donor1 + 0.4*donor2 in ALL periods, then + tau post.
    for p in 0..t {
        let base = 0.6 * y.get(1, p) + 0.4 * y.get(2, p);
        let eff = if p >= t0 { tau } else { 0.0 };
        y.set(0, p, base + eff);
    }

    let panel = Panel::block(y, &[0], t0);
    (panel, t0)
}

#[test]
fn sc_recovers_planted_effect() {
    let tau = 2.5;
    let (panel, _t0) = planted_panel(tau, 11);
    let fit = fit_sc(&panel, ScConfig::default());
    // Pre-fit should be near-perfect (relative error ~1e-7 on values ~10).
    assert!(fit.pre_rmspe < 1e-4, "pre-RMSPE too large: {}", fit.pre_rmspe);
    // ATT should recover tau.
    assert!(
        (fit.att - tau).abs() < 1e-4,
        "ATT {} != tau {}",
        fit.att,
        tau
    );
    // Weights concentrate on donors 1 and 2 (≈0.6, 0.4).
    let id1 = fit.donor_ids.iter().position(|&u| u == 1).unwrap();
    let id2 = fit.donor_ids.iter().position(|&u| u == 2).unwrap();
    assert!((fit.weights[id1] - 0.6).abs() < 1e-2, "w1={}", fit.weights[id1]);
    assert!((fit.weights[id2] - 0.4).abs() < 1e-2, "w2={}", fit.weights[id2]);
}

#[test]
fn sc_zero_effect_is_near_zero() {
    let (panel, _t0) = planted_panel(0.0, 22);
    let fit = fit_sc(&panel, ScConfig::default());
    assert!(fit.att.abs() < 1e-4, "ATT should be ~0, got {}", fit.att);
}

#[test]
fn sc_weights_sum_to_one_and_nonneg() {
    let (panel, _t0) = planted_panel(1.0, 33);
    let fit = fit_sc(&panel, ScConfig::default());
    let sum: f64 = fit.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-8);
    assert!(fit.weights.iter().all(|&w| w >= -1e-9));
}
