//! Correctness gate for the from-scratch numerical core.
//!
//! Strategy: known-answer cases by hand, structural property checks on random
//! matrices, and the SVD cross-oracle (one-sided Jacobi vs the Gram-eig path).

#![allow(clippy::needless_range_loop)]

use panelkit_linalg::factor::cholesky::Cholesky;
use panelkit_linalg::factor::eig_sym::SymEig;
use panelkit_linalg::factor::qr::Qr;
use panelkit_linalg::factor::svd::Svd;
use panelkit_linalg::factor::svd_gram::{singular_values_via_gram, svd_via_gram};
use panelkit_linalg::matrix::Mat;
use panelkit_linalg::ops::matmul::{matmul, matvec};
use panelkit_linalg::ops::norms::frobenius;
use panelkit_linalg::opt::simplex::{project_simplex, sc_weights, solve_fw, solve_pg};
use panelkit_linalg::opt::softthresh::svt;
use panelkit_linalg::rng::Xoshiro256pp;

const TOL: f64 = 1e-9;

fn rand_mat(rng: &mut Xoshiro256pp, r: usize, c: usize) -> Mat {
    let mut m = Mat::zeros(r, c);
    for v in m.as_mut_slice().iter_mut() {
        *v = rng.next_normal();
    }
    m
}

fn diff_frob(a: &Mat, b: &Mat) -> f64 {
    assert_eq!(a.shape(), b.shape());
    let mut acc = 0.0;
    for (x, y) in a.as_slice().iter().zip(b.as_slice().iter()) {
        acc += (x - y).powi(2);
    }
    acc.sqrt()
}

#[test]
fn gemm_known_answer() {
    let a = Mat::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Mat::from_row_major(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = matmul(&a, &b);
    // [1 2 3]·[7 9 11]ᵀcols → [[58,64],[139,154]]
    assert!((c.get(0, 0) - 58.0).abs() < TOL);
    assert!((c.get(0, 1) - 64.0).abs() < TOL);
    assert!((c.get(1, 0) - 139.0).abs() < TOL);
    assert!((c.get(1, 1) - 154.0).abs() < TOL);
}

#[test]
fn cholesky_reconstructs_and_solves() {
    // SPD matrix A = MᵀM + I.
    let mut rng = Xoshiro256pp::seed_from_u64(1);
    let m = rand_mat(&mut rng, 6, 4);
    let mut a = panelkit_linalg::ops::matmul::syrk_ata(&m);
    for i in 0..4 {
        a.add_to(i, i, 1.0);
    }
    let chol = Cholesky::new(&a).unwrap();
    // L Lᵀ ≈ A
    let l = chol.l();
    let lt = l.transpose();
    let recon = matmul(l, &lt);
    assert!(diff_frob(&recon, &a) < 1e-8, "LLᵀ != A");
    // Solve A x = b.
    let xtrue = vec![1.0, -2.0, 0.5, 3.0];
    let b = matvec(&a, &xtrue);
    let x = chol.solve_vec(&b);
    for i in 0..4 {
        assert!((x[i] - xtrue[i]).abs() < 1e-8);
    }
}

#[test]
fn qr_orthonormal_and_reconstructs() {
    let mut rng = Xoshiro256pp::seed_from_u64(2);
    let a = rand_mat(&mut rng, 7, 4);
    let qr = Qr::new(&a).unwrap();
    let q = qr.q_thin();
    let r = qr.r();
    // QᵀQ ≈ I
    let qtq = panelkit_linalg::ops::matmul::syrk_ata(&q);
    let id = Mat::identity(4);
    assert!(diff_frob(&qtq, &id) < 1e-8, "QᵀQ != I");
    // Q R ≈ A
    let recon = matmul(&q, &r);
    assert!(diff_frob(&recon, &a) < 1e-8, "QR != A");
}

#[test]
fn qr_least_squares_matches_normal_equations() {
    let mut rng = Xoshiro256pp::seed_from_u64(3);
    let x = rand_mat(&mut rng, 20, 3);
    let btrue = vec![2.0, -1.0, 0.5];
    let mut y = matvec(&x, &btrue);
    // Add small noise.
    for yi in y.iter_mut() {
        *yi += 0.01 * rng.next_normal();
    }
    let qr = Qr::new(&x).unwrap();
    let bhat = qr.solve_lstsq(&y);
    let bnorm = panelkit_linalg::solve::lstsq::ols_normal(&x, &y).unwrap();
    for i in 0..3 {
        assert!(
            (bhat[i] - bnorm[i]).abs() < 1e-7,
            "QR vs normal-eq disagree"
        );
    }
}

#[test]
fn svd_reconstructs_and_is_orthonormal() {
    let mut rng = Xoshiro256pp::seed_from_u64(4);
    for &(r, c) in &[(8, 5), (5, 8), (6, 6)] {
        let a = rand_mat(&mut rng, r, c);
        let svd = Svd::new(&a);
        let recon = svd.reconstruct();
        assert!(diff_frob(&recon, &a) < 1e-8, "UΣVᵀ != A for {r}x{c}");
        // Orthonormal U, V columns.
        let utu = panelkit_linalg::ops::matmul::syrk_ata(svd.u());
        let vtv = panelkit_linalg::ops::matmul::syrk_ata(svd.v());
        let k = r.min(c);
        let id = Mat::identity(k);
        assert!(diff_frob(&utu, &id) < 1e-8, "UᵀU != I");
        assert!(diff_frob(&vtv, &id) < 1e-8, "VᵀV != I");
        // Non-increasing singular values.
        let s = svd.singular_values();
        for i in 1..s.len() {
            assert!(s[i] <= s[i - 1] + 1e-12);
        }
    }
}

#[test]
fn svd_cross_oracle_singular_values() {
    // The headline correctness test: one-sided Jacobi vs the Gram-eig path.
    let mut rng = Xoshiro256pp::seed_from_u64(5);
    for _ in 0..10 {
        let a = rand_mat(&mut rng, 9, 6);
        let s_jacobi = Svd::new(&a).singular_values().to_vec();
        let s_gram = singular_values_via_gram(&a);
        for i in 0..s_jacobi.len() {
            assert!(
                (s_jacobi[i] - s_gram[i]).abs() < 1e-9,
                "singular value {i} disagrees: {} vs {}",
                s_jacobi[i],
                s_gram[i]
            );
        }
    }
}

#[test]
fn svd_known_diagonal() {
    // A diagonal matrix has its diagonal as singular values (up to sign/order).
    let a = Mat::from_row_major(3, 3, &[3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]);
    let s = Svd::new(&a).singular_values().to_vec();
    assert!((s[0] - 3.0).abs() < TOL);
    assert!((s[1] - 2.0).abs() < TOL);
    assert!((s[2] - 1.0).abs() < TOL);
}

#[test]
fn svd_gram_full_reconstructs() {
    let mut rng = Xoshiro256pp::seed_from_u64(55);
    let a = rand_mat(&mut rng, 7, 4);
    let (u, s, v) = svd_via_gram(&a);
    let recon = u.clone();
    // Build U Σ Vᵀ manually.
    let mut us = u;
    for t in 0..s.len() {
        let col = us.col_mut(t);
        for x in col.iter_mut() {
            *x *= s[t];
        }
    }
    let recon2 = matmul(&us, &v.transpose());
    let _ = recon;
    assert!(diff_frob(&recon2, &a) < 1e-8, "Gram SVD reconstruct failed");
}

#[test]
fn sym_eig_reconstructs() {
    let mut rng = Xoshiro256pp::seed_from_u64(6);
    let m = rand_mat(&mut rng, 5, 5);
    // Symmetrize.
    let mut a = Mat::zeros(5, 5);
    for i in 0..5 {
        for j in 0..5 {
            a.set(i, j, 0.5 * (m.get(i, j) + m.get(j, i)));
        }
    }
    let eig = SymEig::new(&a);
    let v = eig.vectors();
    let vals = eig.values();
    // V Λ Vᵀ ≈ A
    let mut vl = v.clone();
    for t in 0..5 {
        let col = vl.col_mut(t);
        for x in col.iter_mut() {
            *x *= vals[t];
        }
    }
    let recon = matmul(&vl, &v.transpose());
    assert!(diff_frob(&recon, &a) < 1e-8, "VΛVᵀ != A");
}

#[test]
fn sym_eig_known_2x2() {
    // [[2,1],[1,2]] has eigenvalues 3 and 1.
    let a = Mat::from_row_major(2, 2, &[2.0, 1.0, 1.0, 2.0]);
    let eig = SymEig::new(&a);
    let vals = eig.values();
    assert!((vals[0] - 3.0).abs() < TOL);
    assert!((vals[1] - 1.0).abs() < TOL);
}

#[test]
fn simplex_projection_basic() {
    let p = project_simplex(&[0.5, 0.1, 0.4]);
    let sum: f64 = p.iter().sum();
    assert!((sum - 1.0).abs() < TOL);
    assert!(p.iter().all(|&x| x >= -TOL));
    // Already on simplex → unchanged.
    let q = project_simplex(&[0.2, 0.3, 0.5]);
    assert!((q[0] - 0.2).abs() < 1e-9 && (q[2] - 0.5).abs() < 1e-9);
    // Negative entries get clipped, mass redistributed.
    let r = project_simplex(&[-1.0, 2.0, -1.0]);
    assert!((r[1] - 1.0).abs() < 1e-9);
    assert!(r[0].abs() < 1e-9 && r[2].abs() < 1e-9);
}

#[test]
fn sc_weights_recover_planted_convex_combo() {
    // Build donors whose convex combination exactly reproduces the target in the
    // pre-period; SC must recover (close to) the planted weights.
    let mut rng = Xoshiro256pp::seed_from_u64(7);
    let m = 30usize; // pre-periods
    let j = 4usize; // donors
    let y0 = rand_mat(&mut rng, m, j);
    let wtrue = project_simplex(&[0.5, 0.2, 0.0, 0.3]);
    let y = matvec(&y0, &wtrue);
    let sol = sc_weights(&y0, &y, 0.0);
    // Fitted target should match closely.
    let yhat = matvec(&y0, &sol.w);
    let mut err = 0.0;
    for t in 0..m {
        err += (yhat[t] - y[t]).powi(2);
    }
    assert!(
        err.sqrt() < 1e-6,
        "SC pre-fit error too large: {}",
        err.sqrt()
    );
    // Weights on the simplex.
    let sum: f64 = sol.w.iter().sum();
    assert!((sum - 1.0).abs() < 1e-8);
    assert!(sol.w.iter().all(|&x| x >= -1e-9));
}

#[test]
fn simplex_fw_and_pg_agree() {
    let mut rng = Xoshiro256pp::seed_from_u64(8);
    let m = 20usize;
    let j = 5usize;
    let y0 = rand_mat(&mut rng, m, j);
    let y: Vec<f64> = (0..m).map(|_| rng.next_normal()).collect();
    let gram = panelkit_linalg::ops::matmul::syrk_ata(&y0);
    let b = panelkit_linalg::ops::matmul::matvec_t(&y0, &y);
    let fw = solve_fw(&gram, &b, 1e-6, 20000, 1e-12);
    let pg = solve_pg(&gram, &b, 1e-6, 50000, 1e-12);
    // Compare objective values rather than weights (optimum may be on a face).
    let obj = |w: &[f64]| {
        let r = matvec(&y0, w);
        let mut s = 0.0;
        for t in 0..m {
            s += (y[t] - r[t]).powi(2);
        }
        s
    };
    let of = obj(&fw.w);
    let op = obj(&pg.w);
    assert!(
        (of - op).abs() < 1e-5,
        "FW {of} vs PG {op} objective disagree"
    );
}

#[test]
fn svt_thresholds_spectrum() {
    let mut rng = Xoshiro256pp::seed_from_u64(9);
    let a = rand_mat(&mut rng, 6, 6);
    let s = Svd::new(&a).singular_values().to_vec();
    let lambda = s[2]; // threshold between the 3rd and 4th singular value
    let (thr, _nuc) = svt(&a, lambda);
    // The thresholded matrix's singular values must equal max(s - lambda, 0).
    let st = Svd::new(&thr).singular_values().to_vec();
    for i in 0..s.len() {
        let expect = (s[i] - lambda).max(0.0);
        assert!((st[i] - expect).abs() < 1e-7, "SVT spectrum off at {i}");
    }
}

#[test]
fn randomized_svd_recovers_low_rank_spectrum() {
    use panelkit_linalg::factor::randomized::randomized_svd;
    // Build an exactly rank-3 matrix; randomized SVD should match the top-3
    // singular values of the full Jacobi SVD closely.
    let mut rng = Xoshiro256pp::seed_from_u64(101);
    let m = 60;
    let n = 40;
    let r = 3;
    let a_factor = rand_mat(&mut rng, m, r);
    let b_factor = rand_mat(&mut rng, r, n);
    let a = matmul(&a_factor, &b_factor); // rank 3

    let full = Svd::new(&a);
    let rsvd = randomized_svd(&a, r, 8, 2, 42);
    for i in 0..r {
        let rel =
            (rsvd.s[i] - full.singular_values()[i]).abs() / full.singular_values()[i].max(1e-12);
        assert!(
            rel < 1e-6,
            "sv {i}: rand {} vs full {}",
            rsvd.s[i],
            full.singular_values()[i]
        );
    }
    // Reconstruction (no thresholding) recovers A.
    let recon = rsvd.reconstruct_with(&rsvd.s);
    assert!(
        diff_frob(&recon, &a) < 1e-6,
        "randomized reconstruction off"
    );
}

#[test]
fn svt_truncated_matches_full_svt_on_low_rank() {
    use panelkit_linalg::opt::softthresh::{svt, svt_truncated};
    let mut rng = Xoshiro256pp::seed_from_u64(202);
    let a_factor = rand_mat(&mut rng, 50, 4);
    let b_factor = rand_mat(&mut rng, 4, 30);
    let mut a = matmul(&a_factor, &b_factor);
    // add tiny noise so it's near-rank-4
    for v in a.as_mut_slice().iter_mut() {
        *v += 0.001 * rng.next_normal();
    }
    let s = Svd::new(&a).singular_values().to_vec();
    let lambda = s[3] * 0.5;
    let (full_thr, _) = svt(&a, lambda);
    let (trunc_thr, _) = svt_truncated(&a, lambda, 8, 7);
    assert!(
        diff_frob(&full_thr, &trunc_thr) / frobenius(&full_thr) < 1e-3,
        "truncated SVT differs from full SVT"
    );
}

#[test]
fn frobenius_norm_matches_manual() {
    let a = Mat::from_row_major(2, 2, &[3.0, 4.0, 0.0, 0.0]);
    assert!((frobenius(&a) - 5.0).abs() < TOL);
}

// ---------------------------------------------------------------------------
// Scale-robustness regressions: every case below used to return silently
// wrong results (zeroed solutions, identity eigenvectors, wrong singular
// values, oscillating PG iterates) at extreme-but-valid float scales.
// ---------------------------------------------------------------------------

#[test]
fn qr_lstsq_is_scale_invariant_at_tiny_scales() {
    // cond(A) == 1 regardless of units; the rank threshold must be relative.
    // The old absolute 1e-12 floor returned x = 0 here.
    for scale in [1e-13, 1e-30, 1.0, 1e30] {
        let mut a = Mat::zeros(4, 4);
        for i in 0..4 {
            a.set(i, i, scale);
        }
        let b = vec![scale; 4];
        let qr = Qr::new(&a).unwrap();
        let x = qr.solve_lstsq(&b);
        for xi in &x {
            assert!(
                (xi - 1.0).abs() < 1e-10,
                "scale {scale:e}: expected 1.0, got {xi}"
            );
        }
    }
}

#[test]
fn eig_sym_converges_for_tiny_norm_matrices() {
    // ‖A‖ ≈ 3e-16: the old `.max(1.0)` convergence floor accepted the input
    // unrotated (eigenvectors = identity). True pairs: 3e-16 and 1e-16 with
    // ±45° eigenvectors.
    let a = Mat::from_row_major(2, 2, &[2e-16, 1e-16, 1e-16, 2e-16]);
    let eig = SymEig::new(&a);
    let v = eig.values();
    assert!((v[0] - 3e-16).abs() < 1e-22, "λ1 = {}", v[0]);
    assert!((v[1] - 1e-16).abs() < 1e-22, "λ2 = {}", v[1]);
    let vec0 = eig.vectors().col(0);
    assert!(
        (vec0[0].abs() - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-8,
        "eigenvector not rotated: {vec0:?}"
    );
}

#[test]
fn svd_is_scale_invariant_at_extreme_scales() {
    // σ(c·A) = c·σ(A); the raw Gram accumulation used to overflow at 1e200
    // (inf <= inf → "converged") and flush to zero at 1e-180.
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // singular values of [[1,1],[0,1]]
    for scale in [1e200, 1e-180, 1.0] {
        let a = Mat::from_row_major(2, 2, &[scale, scale, 0.0, scale]);
        let s = Svd::new(&a).singular_values().to_vec();
        assert!(
            (s[0] / scale - phi).abs() < 1e-12,
            "scale {scale:e}: σ1/c = {} (want {phi})",
            s[0] / scale
        );
        assert!(
            (s[1] / scale - 1.0 / phi).abs() < 1e-12,
            "scale {scale:e}: σ2/c = {} (want {})",
            s[1] / scale,
            1.0 / phi
        );
    }
}

#[test]
fn householder_qr_survives_huge_columns() {
    // The unscaled column norm overflowed for entries ≳ 1e154 → R = -inf and
    // lstsq returned 0. hypot-based norms keep it exact.
    let a = Mat::from_row_major(2, 1, &[1e200, 1e200]);
    let qr = Qr::new(&a).unwrap();
    let x = qr.solve_lstsq(&[1e200, 1e200]);
    assert!((x[0] - 1.0).abs() < 1e-12, "got {x:?}");
}

#[test]
fn solve_pg_handles_adversarial_gram() {
    // G's top eigenvector [1,-1]/√2 is orthogonal to PG's deterministic
    // power-iteration start (all-ones), so the old Lipschitz estimate was λ₂
    // (a LOWER bound) and the iteration oscillated to w = [0,1] with
    // objective 1.0. True optimum: w* = [31/60, 29/60], objective ≈ 0.19917.
    let gram = Mat::from_row_major(2, 2, &[2.0, -1.0, -1.0, 2.0]);
    let b = vec![0.1, 0.0];
    let pg = solve_pg(&gram, &b, 0.0, 50_000, 1e-12);
    let fw = solve_fw(&gram, &b, 0.0, 50_000, 1e-12);
    for i in 0..2 {
        assert!(
            (pg.w[i] - fw.w[i]).abs() < 1e-6,
            "PG {:?} != FW {:?}",
            pg.w,
            fw.w
        );
    }
    assert!((pg.w[0] - 31.0 / 60.0).abs() < 1e-6, "w = {:?}", pg.w);
}

#[test]
fn nan_input_does_not_panic_sorters() {
    // NaN must propagate as NaN output, not kill the comparator.
    let a = Mat::from_row_major(2, 2, &[1.0, f64::NAN, 0.0, 1.0]);
    let svd = Svd::new(&a);
    assert!(svd.singular_values().iter().any(|s| s.is_nan()));
    let _ = SymEig::new(&a); // must not panic
    let _ = project_simplex(&[f64::NAN, 0.5]); // must not panic
}

#[test]
#[should_panic(expected = "empty range")]
fn gen_range_zero_panics_in_release_too() {
    Xoshiro256pp::seed_from_u64(0).gen_range(0);
}

#[test]
#[should_panic(expected = "rhs length")]
fn cholesky_solve_rejects_wrong_length_rhs() {
    let a = Mat::from_row_major(2, 2, &[4.0, 1.0, 1.0, 3.0]);
    let chol = Cholesky::new(&a).unwrap();
    let _ = chol.solve_vec(&[1.0, 2.0, 77.0, -5.0]);
}

#[test]
#[should_panic(expected = "square")]
fn eig_sym_rejects_non_square_in_release_too() {
    let a = Mat::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let _ = SymEig::new(&a);
}
