//! ASC, SDID and MC-NNM correctness on planted-effect panels.

use panelkit_estimators::mcnnm::{fit_mcnnm, McnnmConfig};
use panelkit_estimators::sc::{fit_asc, fit_sdid, AscConfig, SdidConfig};
use panelkit_estimators::Panel;
use panelkit_linalg::rng::Xoshiro256pp;
use panelkit_linalg::Mat;

/// A low-rank panel: outcomes = factor model (rank 2) + unit/time levels, with a
/// constant additive effect `tau` on the treated unit's post-period.
fn factor_panel(tau: f64, seed: u64, n: usize, t: usize, t0: usize) -> Panel {
    let mut rng = Xoshiro256pp::seed_from_u64(seed);
    let r = 2usize;
    // Unit factors (n×r) and time factors (t×r).
    let uf: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..r).map(|_| rng.next_normal()).collect())
        .collect();
    let tf: Vec<Vec<f64>> = (0..t)
        .map(|_| (0..r).map(|_| rng.next_normal()).collect())
        .collect();
    let unit_level: Vec<f64> = (0..n).map(|_| 5.0 + rng.next_normal()).collect();
    let time_level: Vec<f64> = (0..t).map(|_| 2.0 + 0.5 * rng.next_normal()).collect();

    let mut y = Mat::zeros(n, t);
    for i in 0..n {
        for p in 0..t {
            let mut v = unit_level[i] + time_level[p];
            for k in 0..r {
                v += uf[i][k] * tf[p][k];
            }
            if i == 0 && p >= t0 {
                v += tau;
            }
            y.set(i, p, v);
        }
    }
    Panel::block(y, &[0], t0)
}

#[test]
fn asc_recovers_effect_on_clean_panel() {
    // ASC should recover tau well when donors can match the treated pre-path.
    let tau = 2.0;
    let panel = factor_panel(tau, 101, 12, 30, 22);
    let fit = fit_asc(&panel, AscConfig::default());
    assert!(
        (fit.att - tau).abs() < 0.5,
        "ASC att {} far from tau {}",
        fit.att,
        tau
    );
}

#[test]
fn sdid_recovers_effect() {
    let tau = 2.0;
    let panel = factor_panel(tau, 202, 16, 30, 22);
    let fit = fit_sdid(&panel, SdidConfig::default());
    assert!(
        (fit.att - tau).abs() < 0.5,
        "SDID att {} far from tau {}",
        fit.att,
        tau
    );
    // Unit weights are a valid simplex.
    let sum: f64 = fit.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(fit.weights.iter().all(|&w| w >= -1e-9));
}

#[test]
fn sdid_zero_effect_near_zero() {
    let panel = factor_panel(0.0, 303, 16, 30, 22);
    let fit = fit_sdid(&panel, SdidConfig::default());
    assert!(
        fit.att.abs() < 0.5,
        "SDID att should be ~0, got {}",
        fit.att
    );
}

#[test]
fn mcnnm_recovers_effect_on_low_rank_panel() {
    // MC-NNM is designed exactly for the low-rank DGP.
    let tau = 3.0;
    let panel = factor_panel(tau, 404, 20, 30, 24);
    let fit = fit_mcnnm(&panel, McnnmConfig::default());
    assert!(
        (fit.att - tau).abs() < 0.8,
        "MC-NNM att {} far from tau {}",
        fit.att,
        tau
    );
}

#[test]
fn mcnnm_zero_effect_near_zero() {
    let panel = factor_panel(0.0, 505, 20, 30, 24);
    let fit = fit_mcnnm(&panel, McnnmConfig::default());
    assert!(
        fit.att.abs() < 0.8,
        "MC-NNM att should be ~0, got {}",
        fit.att
    );
}

#[test]
fn asc_is_translation_invariant() {
    // BMFR's ridge augmentation includes an intercept (the ridge is fitted on
    // donor-centered data), so adding a constant to every outcome must not
    // change the ATT. The old uncentered Gram failed this: the same panel
    // shifted by +1000 gave a different ATT.
    let tau = 2.0;
    let base = factor_panel(tau, 707, 12, 30, 22);
    for shift in [100.0, 1e4] {
        let mut y = base.y().clone();
        for v in y.as_mut_slice().iter_mut() {
            *v += shift;
        }
        let shifted = Panel::block(y, &[0], 22);
        for cfg in [
            AscConfig::default(),
            AscConfig {
                sc_ridge: 0.0,
                aug_lambda: Some(10.0),
            },
        ] {
            let a = fit_asc(&base, cfg).att;
            let b = fit_asc(&shifted, cfg).att;
            assert!(
                (a - b).abs() < 1e-6,
                "ASC not translation-invariant: {a} vs {b} (shift {shift})"
            );
        }
    }
}

#[test]
fn asc_constant_donors_falls_back_to_sc() {
    // Zero cross-sectional donor variation → centered Gram ≡ 0 → augmentation
    // has nothing to learn; must not panic, and must equal plain SC.
    let n = 5;
    let t = 12;
    let mut y = Mat::zeros(n, t);
    for i in 0..n {
        for p in 0..t {
            // All donors identical; treated unit shifted.
            y.set(i, p, if i == 0 { 10.0 } else { 7.0 });
        }
    }
    let panel = Panel::block(y, &[0], 8);
    let asc = fit_asc(&panel, AscConfig::default());
    let sc = panelkit_estimators::sc::fit_sc(&panel, panelkit_estimators::sc::ScConfig::default());
    assert!(
        (asc.att - sc.att).abs() < 1e-10,
        "constant-donor ASC {} != SC {}",
        asc.att,
        sc.att
    );
}
