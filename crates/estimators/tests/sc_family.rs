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
