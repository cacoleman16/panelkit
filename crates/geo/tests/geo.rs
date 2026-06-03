//! Tests for the geo design engine: power monotonicity, MDE sanity, diagnostics,
//! and market selection.

use panelkit_geo::power::power_curve;
use panelkit_geo::selection::{select_markets, SelectConfig};
use panelkit_geo::types::Method;
use panelkit_geo::{diagnostics, Diagnostics};
use panelkit_linalg::rng::Xoshiro256pp;
use panelkit_linalg::Mat;

/// A clean factor-model geo panel: N markets, T periods, no real treatment.
/// Market 0 is a tight convex mix of two donors (so SC fits it well).
fn geo_panel(n: usize, t: usize, seed: u64) -> Mat {
    let mut rng = Xoshiro256pp::seed_from_u64(seed);
    let mut y = Mat::zeros(n, t);
    for u in 1..n {
        let mut level = 5.0 + rng.next_normal();
        for p in 0..t {
            level += 0.2 * rng.next_normal();
            y.set(u, p, 50.0 + 10.0 * level + 3.0 * (u as f64));
        }
    }
    for p in 0..t {
        let base = 0.5 * y.get(1, p) + 0.5 * y.get(2, p);
        y.set(0, p, base + 0.5 * rng.next_normal());
    }
    y
}

#[test]
fn power_increases_with_lift_and_mde_is_sane() {
    let y = geo_panel(15, 60, 1);
    let lifts = vec![0.0, 0.02, 0.05, 0.10, 0.20];
    let pr = power_curve(&y, &[0], 10, &lifts, Method::Sc, 0.10, 0.8, 20);

    // Power is (weakly) increasing in lift.
    for w in pr.points.windows(2) {
        assert!(
            w[1].power >= w[0].power - 0.2,
            "power should rise with lift: {:?}",
            pr.points
        );
    }
    // Null power (lift 0) is near the nominal alpha-ish level, not huge.
    assert!(
        pr.points[0].power <= 0.4,
        "null power too high: {}",
        pr.points[0].power
    );
    // Big lift is detected often.
    assert!(
        pr.points.last().unwrap().power >= 0.6,
        "20% lift should be detectable"
    );
    // MDE, if found, is within the grid range.
    if let Some(m) = pr.mde_pct {
        assert!(m > 0.0 && m <= 0.20 + 1e-9);
        assert!(pr.mde_abs_per_period.unwrap() > 0.0);
        assert!(pr.mde_cumulative.unwrap() > 0.0);
    }
}

#[test]
fn estimated_lift_tracks_true_lift() {
    let y = geo_panel(15, 60, 2);
    let lifts = vec![0.0, 0.10];
    let pr = power_curve(&y, &[0], 10, &lifts, Method::Sc, 0.10, 0.8, 20);
    // At a 10% injected lift, the mean estimated lift should be in the ballpark.
    let p10 = pr.points.last().unwrap();
    assert!(
        (p10.est_pct_mean - 0.10).abs() < 0.05,
        "estimated lift {} far from 10%",
        p10.est_pct_mean
    );
}

#[test]
fn diagnostics_reasonable_on_clean_panel() {
    let y = geo_panel(15, 60, 3);
    let d: Diagnostics = diagnostics(&y, &[0], 10);
    assert!(d.holdout_pct > 0.0 && d.holdout_pct < 1.0);
    assert!(d.confidence >= 0.0 && d.confidence <= 100.0);
    assert!(d.stability_score >= 0.0 && d.stability_score <= 1.0);
    // Market 0 is an exact donor mix → SC should beat naive DiD.
    assert!(d.improvement_vs_naive >= 0.0);
}

#[test]
fn all_three_methods_run() {
    let y = geo_panel(15, 60, 4);
    let lifts = vec![0.0, 0.10];
    for m in [Method::Sc, Method::Asc, Method::Sdid] {
        let pr = power_curve(&y, &[0], 10, &lifts, m, 0.10, 0.8, 20);
        assert_eq!(pr.method, m);
        assert_eq!(pr.points.len(), 2);
    }
}

#[test]
fn market_selection_ranks_candidates() {
    let y = geo_panel(12, 60, 5);
    let cfg = SelectConfig {
        eligible: (0..12).collect(),
        max_treated: 3,
        test_len: 10,
        target_lift: 0.10,
        method: Method::Sc,
        alpha: 0.10,
        target_power: 0.8,
        min_pre: 20,
        n_candidates: 20,
        seed: 7,
    };
    let ranked = select_markets(&y, &cfg);
    assert!(!ranked.is_empty());
    // Sorted descending by score.
    for w in ranked.windows(2) {
        assert!(w[0].score >= w[1].score - 1e-12);
    }
    // Every candidate has a valid holdout and confidence.
    for c in &ranked {
        assert!(c.holdout_pct > 0.0 && c.holdout_pct < 1.0);
        assert!(c.confidence >= 0.0 && c.confidence <= 100.0);
    }
}
