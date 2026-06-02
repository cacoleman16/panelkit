//! CP-ASC family correctness: pooled recovery across multiple treated units,
//! all three pooling modes, and conformal-inference behaviour (small p-value
//! under a real effect, large under the null).

use panelkit_estimators::sc::cpasc::{fit_at, CpascConfig, PoolMode};
use panelkit_estimators::Panel;
use panelkit_linalg::rng::Xoshiro256pp;
use panelkit_linalg::Mat;

/// Build a multi-treated panel: `n_treated` treated units (each an exact convex
/// combo of two donors in the pre-period) plus donors. A constant effect `tau`
/// turns on post-treatment for the treated units.
fn multi_treated_panel(n_treated: usize, tau: f64, seed: u64) -> (Panel, usize) {
    let mut rng = Xoshiro256pp::seed_from_u64(seed);
    let n_donor = 8usize;
    let n = n_treated + n_donor;
    let t = 30usize;
    let t0 = 22usize;

    let mut y = Mat::zeros(n, t);
    // Donors are random walks (units n_treated..n).
    for u in n_treated..n {
        let mut level = rng.next_normal();
        for p in 0..t {
            level += 0.3 * rng.next_normal();
            y.set(u, p, 10.0 + level + 0.5 * (u as f64));
        }
    }
    // Treated units 0..n_treated: convex combo of the first two donors + noise.
    let d1 = n_treated;
    let d2 = n_treated + 1;
    let mut treated = Vec::new();
    for u in 0..n_treated {
        let a = 0.4 + 0.1 * (u as f64); // varied mixing per unit
        for p in 0..t {
            let base = a * y.get(d1, p) + (1.0 - a) * y.get(d2, p);
            let eff = if p >= t0 { tau } else { 0.0 };
            y.set(u, p, base + eff + 0.05 * rng.next_normal());
        }
        treated.push(u);
    }
    (Panel::block(y, &treated, t0), t0)
}

#[test]
fn cpasc_mspe_recovers_pooled_effect() {
    let tau = 2.0;
    let (panel, t0) = multi_treated_panel(6, tau, 1);
    let cfg = CpascConfig {
        mode: PoolMode::Mspe,
        ..Default::default()
    };
    let fit = fit_at(&panel, t0, cfg);
    assert!(
        (fit.att - tau).abs() < 0.4,
        "CP-ASC att {} far from tau {}",
        fit.att,
        tau
    );
    // Weights sum to 1.
    let wsum: f64 = fit.units.iter().map(|u| u.weight).sum();
    assert!((wsum - 1.0).abs() < 1e-9);
    // Real effect → small conformal p-value.
    assert!(fit.p_value < 0.2, "expected small p, got {}", fit.p_value);
}

#[test]
fn cpasc_null_has_large_pvalue() {
    let (panel, t0) = multi_treated_panel(6, 0.0, 2);
    let cfg = CpascConfig::default();
    let fit = fit_at(&panel, t0, cfg);
    assert!(
        fit.att.abs() < 0.4,
        "null att should be ~0, got {}",
        fit.att
    );
    // No effect → conformal p-value not significant at the 5% level.
    assert!(
        fit.p_value > 0.05,
        "expected non-significant p under null, got {}",
        fit.p_value
    );
}

#[test]
fn stratified_and_cumulative_modes_run_and_pool() {
    let tau = 2.0;
    let (panel, t0) = multi_treated_panel(8, tau, 3);
    for mode in [PoolMode::Stratified { n_strata: 3 }, PoolMode::Cumulative] {
        let cfg = CpascConfig {
            mode,
            ..Default::default()
        };
        let fit = fit_at(&panel, t0, cfg);
        let wsum: f64 = fit.units.iter().map(|u| u.weight).sum();
        assert!((wsum - 1.0).abs() < 1e-9, "{:?} weights sum {}", mode, wsum);
        assert!(
            (fit.att - tau).abs() < 0.6,
            "{:?} att {} far from tau {}",
            mode,
            fit.att,
            tau
        );
    }
}
