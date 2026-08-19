//! SC-family correctness on planted-effect panels: ASC, SDID, MC-NNM, the
//! Ferman-Pinto demeaned estimator, robust SC, and the donor weight bounds.

use panelkit_estimators::mcnnm::{fit_mcnnm, McnnmConfig};
use panelkit_estimators::sc::{
    fit_asc, fit_fp, fit_rsc, fit_sc, fit_sdid, AscConfig, FpConfig, RscConfig, ScConfig,
    SdidConfig,
};
use panelkit_estimators::Panel;
use panelkit_linalg::opt::simplex::WeightBounds;
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
                aug_lambda: Some(10.0),
                ..AscConfig::default()
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

// ---------------------------------------------------------------------------
// Demeaned SC (Ferman-Pinto), robust SC, penalized SC, and donor weight bounds.
// ---------------------------------------------------------------------------

/// `factor_panel` with the treated unit's level pushed `gap` above every donor,
/// so no convex combination of donors can match its pre-treatment level.
fn level_gap_panel(tau: f64, gap: f64, seed: u64, n: usize, t: usize, t0: usize) -> Panel {
    let panel = factor_panel(tau, seed, n, t, t0);
    let mut y = panel.y().clone();
    for p in 0..t {
        y.set(0, p, y.get(0, p) + gap);
    }
    Panel::block(y, &[0], t0)
}

#[test]
fn demeaned_sc_survives_a_treated_level_outside_the_donor_hull() {
    // The Ferman-Pinto case: classic SC spends its weights trying to close a
    // level gap it cannot close, and the unclosed remainder lands in the ATT.
    // Demeaning removes the level from the matching problem entirely.
    let (tau, gap) = (2.0, 40.0);
    let panel = level_gap_panel(tau, gap, 77, 14, 30, 22);
    let sc = fit_sc(&panel, ScConfig::default());
    let fp = fit_fp(&panel, FpConfig::default());
    assert!(
        (fp.att - tau).abs() < 0.5,
        "demeaned SC att {} far from tau {tau}",
        fp.att
    );
    assert!(
        (sc.att - tau).abs() > 5.0 * (fp.att - tau).abs(),
        "expected classic SC to be badly biased here (sc {} vs fp {})",
        sc.att,
        fp.att
    );
    // Weights stay a convex combination.
    let sum: f64 = fp.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6 && fp.weights.iter().all(|&w| w >= -1e-9));
}

#[test]
fn demeaned_sc_is_invariant_to_a_treated_level_shift() {
    // Shifting the treated unit by a constant must not move the ATT at all —
    // that invariance *is* the estimator.
    let panel = factor_panel(1.5, 909, 12, 30, 22);
    let base = fit_fp(&panel, FpConfig::default()).att;
    let mut y = panel.y().clone();
    for p in 0..y.cols() {
        y.set(0, p, y.get(0, p) + 123.456);
    }
    let shifted = fit_fp(&Panel::block(y, &[0], 22), FpConfig::default()).att;
    assert!(
        (base - shifted).abs() < 1e-8,
        "demeaned SC moved with a level shift: {base} vs {shifted}"
    );
}

#[test]
fn robust_sc_recovers_effect_on_a_low_rank_panel() {
    let tau = 2.0;
    let panel = factor_panel(tau, 313, 16, 30, 22);
    let fit = fit_rsc(&panel, RscConfig::default());
    assert!(
        (fit.att - tau).abs() < 0.5,
        "robust SC att {} far from tau {tau}",
        fit.att
    );
}

#[test]
fn robust_sc_extrapolates_beyond_the_donor_hull() {
    // Unconstrained weights are the point of this estimator: with the treated
    // unit scaled beyond every donor, a convex combination cannot reach it but a
    // linear one can.
    let t0 = 22usize;
    let panel = factor_panel(0.0, 555, 14, 30, t0);
    let mut y = panel.y().clone();
    let tau = 3.0;
    for p in 0..y.cols() {
        // Treated unit = 2.5x the factor structure of donor 1 (outside the hull).
        let v = 2.5 * y.get(1, p);
        y.set(0, p, if p >= t0 { v + tau } else { v });
    }
    let fit = fit_rsc(&Panel::block(y, &[0], t0), RscConfig::default());
    assert!(
        (fit.att - tau).abs() < 0.5,
        "robust SC att {} far from tau {tau} on an out-of-hull treated unit",
        fit.att
    );
}

#[test]
fn donor_weight_bounds_are_enforced_across_the_family() {
    let panel = factor_panel(2.0, 4242, 20, 40, 30);
    let cap = 0.15;
    let floor = 0.01;
    let bounds = WeightBounds::new(floor, cap);
    let sc = fit_sc(
        &panel,
        ScConfig {
            bounds,
            ..ScConfig::default()
        },
    );
    let asc = fit_asc(
        &panel,
        AscConfig {
            bounds,
            ..AscConfig::default()
        },
    );
    let sdid = fit_sdid(
        &panel,
        SdidConfig {
            bounds,
            ..SdidConfig::default()
        },
    );
    let fp = fit_fp(&panel, FpConfig { ridge: 0.0, bounds });
    for (name, fit) in [("sc", &sc), ("asc", &asc), ("sdid", &sdid), ("fp", &fp)] {
        let sum: f64 = fit.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "{name}: weights sum to {sum}");
        for &w in &fit.weights {
            assert!(
                w >= floor - 1e-7 && w <= cap + 1e-7,
                "{name}: weight {w} outside [{floor}, {cap}]"
            );
        }
    }
    // The cap must actually bind: unbounded SC concentrates more than 15%.
    let free = fit_sc(&panel, ScConfig::default());
    assert!(
        free.weights.iter().cloned().fold(0.0, f64::max) > cap,
        "test panel does not exercise the cap"
    );
}

#[test]
fn penalized_sc_prefers_donors_that_resemble_the_treated_unit() {
    // Abadie & L'Hour: among equally-good fits, the penalty picks the donors
    // that are individually close to the treated unit.
    let panel = factor_panel(0.0, 8181, 18, 40, 30);
    let t0 = 30usize;
    let (z0, _) = panel.donor_pre(t0);
    let treated_pre = panel.unit_mean(&[0])[..t0].to_vec();
    let dist2 = |w: &[f64]| -> f64 {
        (0..z0.cols())
            .map(|j| {
                let d: f64 = (0..t0)
                    .map(|t| (treated_pre[t] - z0.get(t, j)).powi(2))
                    .sum();
                w[j] * d
            })
            .sum::<f64>()
    };
    let plain = fit_sc(&panel, ScConfig::default());
    let pen = fit_sc(
        &panel,
        ScConfig {
            penalty: 0.5,
            ..ScConfig::default()
        },
    );
    assert!(
        dist2(&pen.weights) < dist2(&plain.weights),
        "penalty did not shift weight toward closer donors: {} vs {}",
        dist2(&pen.weights),
        dist2(&plain.weights)
    );
    let sum: f64 = pen.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6 && pen.weights.iter().all(|&w| w >= -1e-9));
}
