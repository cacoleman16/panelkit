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

#[test]
fn mcnnm_is_level_shift_invariant() {
    // The nuclear-norm penalty must not shrink the level component: the
    // unpenalized two-way fixed effects absorb it (Athey et al. include Γ, Δ
    // unpenalized). Pre-fix, shifting the panel by +1000 took the ATT from
    // ~3 to ~16 on this DGP.
    let tau = 3.0;
    let base = factor_panel(tau, 404, 20, 30, 24);
    let fit0 = fit_mcnnm(&base, McnnmConfig::default());
    let mut y = base.y().clone();
    for v in y.as_mut_slice().iter_mut() {
        *v += 1000.0;
    }
    let shifted = Panel::block(y, &[0], 24);
    let fit1 = fit_mcnnm(&shifted, McnnmConfig::default());
    assert!(
        (fit0.att - fit1.att).abs() < 0.25,
        "MC-NNM level-shift drift: {} vs {}",
        fit0.att,
        fit1.att
    );
    assert!(
        (fit1.att - tau).abs() < 0.8,
        "MC-NNM att {} far from tau {} on shifted panel",
        fit1.att,
        tau
    );
}

#[test]
fn mcnnm_exact_on_two_way_additive_panel() {
    // Pure unit + time structure (rank-0 residual): the FE terms should do all
    // the work and recover tau almost exactly, at any outcome level.
    let (n, t, t0) = (12, 20, 14);
    let tau = 2.5;
    let mut rng = Xoshiro256pp::seed_from_u64(606);
    let a: Vec<f64> = (0..n)
        .map(|_| 10_000.0 + 50.0 * rng.next_normal())
        .collect();
    let b: Vec<f64> = (0..t).map(|_| 20.0 * rng.next_normal()).collect();
    let mut y = Mat::zeros(n, t);
    for (i, ai) in a.iter().enumerate() {
        for (p, bp) in b.iter().enumerate() {
            let mut v = ai + bp;
            if i == 0 && p >= t0 {
                v += tau;
            }
            y.set(i, p, v);
        }
    }
    let fit = fit_mcnnm(&Panel::block(y, &[0], t0), McnnmConfig::default());
    assert!(
        (fit.att - tau).abs() < 0.05,
        "MC-NNM att {} should be ~{} on an additive panel",
        fit.att,
        tau
    );
}

#[test]
fn mcnnm_tiny_lambda_is_not_the_zero_fill() {
    // Cold-started SoftImpute at λ ≈ 0 used to hit a trivial fixed point where
    // every missing cell stayed at its zero fill ("counterfactual = 0", ATT =
    // the raw treated level). The warm-started continuation path must not.
    let tau = 3.0;
    let panel = factor_panel(tau, 404, 20, 30, 24);
    let cfg = McnnmConfig {
        lambda: Some(1e-9),
        ..McnnmConfig::default()
    };
    let fit = fit_mcnnm(&panel, cfg);
    let cf_norm: f64 = fit.counterfactual_post.iter().map(|v| v.abs()).sum();
    assert!(
        cf_norm > 1.0,
        "counterfactual collapsed to the zero fill: {:?}",
        fit.counterfactual_post
    );
    // λ→0 overfits, but the answer must stay in a sane neighborhood of tau —
    // not equal to the raw treated post level (~7 on this DGP).
    assert!(
        (fit.att - tau).abs() < 2.0,
        "tiny-lambda ATT {} not in a sane neighborhood of {}",
        fit.att,
        tau
    );
}
