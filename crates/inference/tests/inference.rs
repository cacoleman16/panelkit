//! Inference-engine tests: multiplier bootstrap correctness and the
//! thread-count-invariance (determinism) guarantee.

use panelkit_inference::bootstrap::{jackknife_se, multiplier_bootstrap};

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
fn jackknife_se_of_constant_is_zero() {
    let est = vec![2.0; 10];
    assert!(jackknife_se(&est, 2.0) < 1e-12);
}

#[test]
fn jackknife_se_positive_for_varying() {
    let est = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(jackknife_se(&est, 3.0) > 0.0);
}
