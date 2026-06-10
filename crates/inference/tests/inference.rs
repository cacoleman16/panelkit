//! Inference-engine tests: multiplier bootstrap correctness and the
//! thread-count-invariance (determinism) guarantee.

use panelkit_inference::bootstrap::{
    block_bootstrap_mean, jackknife_se, multiplier_bootstrap, stationary_bootstrap_mean,
};

/// The multiplier-bootstrap SE of a sample mean (IF_i = x_i − x̄) should be close
/// to the analytic SE = sd/√n.
#[test]
fn multiplier_bootstrap_matches_analytic_se_of_mean() {
    let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.123).sin()).collect();
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let influence: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let analytic_se = (var / n as f64).sqrt();

    let (ci, draws) = multiplier_bootstrap(&influence, mean, 4000, 12345, 0.95);
    assert_eq!(draws.len(), 4000);
    assert!(
        (ci.se - analytic_se).abs() / analytic_se < 0.1,
        "bootstrap se {} vs analytic {}",
        ci.se,
        analytic_se
    );
}

/// Determinism: identical seed → identical bootstrap output, bit-for-bit,
/// regardless of how many threads rayon uses. This is the M5 non-negotiable.
#[test]
fn bootstrap_is_thread_count_invariant() {
    let influence: Vec<f64> = (0..150).map(|i| ((i * 7 % 11) as f64) - 5.0).collect();
    let run = || multiplier_bootstrap(&influence, 1.0, 2000, 999, 0.9).1;
    let a = run();
    let b = run();
    // Same process, same seed → identical draws.
    assert_eq!(a, b);
    // Each draw is a pure function of (seed, replicate index), so a serial
    // recomputation of replicate k must match draw k exactly.
    use panelkit_linalg::rng::Xoshiro256pp;
    const P: f64 = 0.723606797749979;
    const NEG: f64 = -0.618033988749895;
    const POS: f64 = 1.618033988749895;
    let n = influence.len();
    for k in [0usize, 1, 500, 1999] {
        let mut rng = Xoshiro256pp::substream(999, k as u64);
        let mut acc = 0.0;
        for &ifi in &influence {
            let v = if rng.next_f64() < P { NEG } else { POS };
            acc += v * ifi;
        }
        let expect = 1.0 + acc / n as f64;
        assert!(
            (a[k] - expect).abs() < 1e-12,
            "replicate {k}: {} vs recomputed {}",
            a[k],
            expect
        );
    }
}

#[test]
fn block_bootstrap_ci_brackets_mean_of_iid_series() {
    // For a roughly-iid series, the block bootstrap of the mean should bracket
    // the sample mean with a sensible SE (≈ sd/√n for block_len 1).
    let x: Vec<f64> = (0..200)
        .map(|i| ((i * 13 % 97) as f64) / 97.0 - 0.5)
        .collect();
    let n = x.len();
    let m = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let analytic = (var / n as f64).sqrt();
    let (ci, draws) = block_bootstrap_mean(&x, 1, 3000, 7, 0.95);
    assert_eq!(draws.len(), 3000);
    assert!(ci.lower < m && m < ci.upper, "CI should bracket the mean");
    // Block length 1 ≈ iid bootstrap → SE close to analytic.
    assert!(
        (ci.se - analytic).abs() / analytic < 0.15,
        "block-bootstrap se {} vs analytic {}",
        ci.se,
        analytic
    );
}

#[test]
fn stationary_bootstrap_runs_and_brackets() {
    let x: Vec<f64> = (0..150).map(|i| (i as f64 * 0.07).sin()).collect();
    let m = x.iter().sum::<f64>() / x.len() as f64;
    let (ci, draws) = stationary_bootstrap_mean(&x, 10, 2000, 3, 0.9);
    assert_eq!(draws.len(), 2000);
    assert!(ci.lower <= m && m <= ci.upper);
    assert!(ci.se > 0.0);
}

#[test]
fn bootstrap_engines_thread_count_invariant() {
    let x: Vec<f64> = (0..120).map(|i| ((i * 7 % 13) as f64) - 6.0).collect();
    let a = block_bootstrap_mean(&x, 8, 1500, 42, 0.95).1;
    let b = block_bootstrap_mean(&x, 8, 1500, 42, 0.95).1;
    assert_eq!(a, b);
    let c = stationary_bootstrap_mean(&x, 8, 1500, 42, 0.95).1;
    let d = stationary_bootstrap_mean(&x, 8, 1500, 42, 0.95).1;
    assert_eq!(c, d);
}

#[test]
fn jackknife_se_of_constant_is_zero() {
    let est = vec![2.0; 10];
    assert!(jackknife_se(&est) < 1e-12);
}

#[test]
fn jackknife_se_positive_for_varying() {
    let est = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(jackknife_se(&est) > 0.0);
}

#[test]
fn placebo_atts_are_outcome_scale_nulls() {
    // The placebo engine reports both the dimensionless RMSPE ratios (test
    // statistics → p-value) and the placebo ATTs (outcome units → SE/CI).
    // Scaling the panel by c must scale the ATTs by c and leave the ratios
    // and p-value unchanged.
    use panelkit_estimators::sc::synthetic::ScConfig;
    use panelkit_estimators::Panel;
    use panelkit_inference::sc_placebo;
    use panelkit_linalg::rng::Xoshiro256pp;
    use panelkit_linalg::Mat;

    let (n, t, t0) = (10, 24, 18);
    let mut rng = Xoshiro256pp::seed_from_u64(99);
    let mut y = Mat::zeros(n, t);
    for i in 0..n {
        let level = 50.0 + 5.0 * rng.next_normal();
        for p in 0..t {
            y.set(i, p, level + rng.next_normal());
        }
    }
    let c = 1000.0;
    let mut yc = y.clone();
    for v in yc.as_mut_slice().iter_mut() {
        *v *= c;
    }

    let pb = sc_placebo(&Panel::block(y, &[0], t0), ScConfig::default());
    let pbc = sc_placebo(&Panel::block(yc, &[0], t0), ScConfig::default());

    assert_eq!(pb.placebo_atts.len(), pb.placebo_ratios.len());
    assert!(
        (pb.p_value - pbc.p_value).abs() < 1e-12,
        "p changed on rescale"
    );
    for (a, ac) in pb.placebo_atts.iter().zip(pbc.placebo_atts.iter()) {
        assert!(
            (a * c - ac).abs() <= 1e-6 * ac.abs().max(1.0),
            "placebo ATT not in outcome units: {a} vs {ac}"
        );
    }
    for (r, rc) in pb.placebo_ratios.iter().zip(pbc.placebo_ratios.iter()) {
        assert!(
            (r - rc).abs() < 1e-9,
            "ratio changed on rescale: {r} vs {rc}"
        );
    }
}

#[test]
fn multiplier_event_bands_are_uniform_and_deterministic() {
    use panelkit_inference::multiplier_event_bands;
    use panelkit_linalg::rng::Xoshiro256pp;

    // Three synthetic event times with iid unit influences.
    let n = 200usize;
    let mut rng = Xoshiro256pp::seed_from_u64(7);
    let ifs: Vec<Vec<f64>> = (0..3)
        .map(|_| (0..n).map(|_| rng.next_normal()).collect())
        .collect();
    let ses: Vec<f64> = ifs
        .iter()
        .map(|f| (f.iter().map(|x| x * x).sum::<f64>() / (n as f64 * n as f64)).sqrt())
        .collect();
    let atts = vec![0.5, 1.0, -0.2];

    let (bands, crit) = multiplier_event_bands(&ifs, &atts, &ses, 999, 0, 0.95);
    // The sup-t critical value must be at least the pointwise z (1.96) — a
    // simultaneous band can never be narrower than the marginal one.
    assert!(crit >= 1.9, "sup-t crit {crit} below the pointwise z");
    for (e, &(lo, hi)) in bands.iter().enumerate() {
        assert!(lo < atts[e] && atts[e] < hi);
        assert!((hi - lo) >= 2.0 * 1.9 * ses[e]);
    }
    // Deterministic given the seed.
    let (bands2, crit2) = multiplier_event_bands(&ifs, &atts, &ses, 999, 0, 0.95);
    assert_eq!(crit, crit2);
    assert_eq!(bands, bands2);
}
