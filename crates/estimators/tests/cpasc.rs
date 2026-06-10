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

#[test]
fn conformal_pvalue_is_size_controlled_under_null() {
    // 200 seeded replications of the package's own multi-treated DGP with
    // tau = 0: P(p <= 0.10) must sit near/below the nominal 10% (the block
    // permutation has granularity 1/T = 1/30 here, so uniform-null mass at or
    // below 0.10 is exactly 3/30 = 0.10). The old pre/post-asymmetric residual
    // construction rejected at ~2x nominal on this DGP — precisely because the
    // per-unit fits are good here (in-hull treated units).
    let n_rep = 200usize;
    let mut reject = 0usize;
    for rep in 0..n_rep {
        let (panel, t0) = multi_treated_panel(4, 0.0, 1_000 + rep as u64);
        let fit = fit_at(&panel, t0, CpascConfig::default());
        if fit.p_value <= 0.10 {
            reject += 1;
        }
    }
    let rate = reject as f64 / n_rep as f64;
    // 0.16 = nominal 0.10 + ~3 MC standard errors at n=200.
    assert!(
        rate <= 0.16,
        "conformal test anti-conservative under the null: P(p<=0.1) = {rate}"
    );
}

#[test]
fn conformal_pvalue_keeps_power_under_real_effect() {
    // A valid permutation test pays for size with power: the null-imposed
    // refit absorbs part of a persistent effect into the weights, and circular
    // blocks overlapping the post window compete with it, so power rises
    // slowly in tau (measured on this DGP: ~0.57 at tau=2, ~0.92 at tau=6).
    // This is regression protection that a clear effect stays detectable —
    // the replications are seeded, so the rate is deterministic.
    let n_rep = 100usize;
    let mut detected = 0usize;
    for rep in 0..n_rep {
        let (panel, t0) = multi_treated_panel(4, 6.0, 9_000 + rep as u64);
        let fit = fit_at(&panel, t0, CpascConfig::default());
        if fit.p_value <= 0.10 {
            detected += 1;
        }
    }
    let rate = detected as f64 / n_rep as f64;
    assert!(
        rate >= 0.85,
        "conformal test lost its power under tau=6: detection rate {rate}"
    );
}
