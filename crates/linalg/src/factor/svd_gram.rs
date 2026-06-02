//! An independent SVD path via the eigendecomposition of the Gram matrix —
//! used as a **cross-oracle** to validate the primary one-sided Jacobi SVD.
//!
//! This is a genuinely different algorithm (full two-sided symmetric Jacobi on
//! `AᵀA` or `A Aᵀ`, then back out `U`/`V`) than the column-rotation Hestenes
//! method in [`super::svd`], so agreement between the two to ~1e-10 on random
//! matrices is strong evidence both are correct. It is intentionally not the
//! production path: forming the Gram matrix squares the condition number and
//! loses relative accuracy on tiny singular values.

use crate::factor::eig_sym::SymEig;
use crate::matrix::Mat;
use crate::ops::matmul::matmul;
use crate::ops::norms::nrm2;

/// Singular values only, via the smaller Gram matrix. Returns them
/// non-increasing, length `min(m, n)`.
pub fn singular_values_via_gram(a: &Mat) -> Vec<f64> {
    let (m, n) = a.shape();
    let gram = if m >= n {
        // AᵀA is n×n
        crate::ops::matmul::syrk_ata(a)
    } else {
        crate::ops::matmul::syrk_aat(a)
    };
    let eig = SymEig::new(&gram);
    let k = m.min(n);
    eig.values()
        .iter()
        .take(k)
        .map(|&lam| lam.max(0.0).sqrt())
        .collect()
}

/// Full thin SVD via the Gram route, returned as `(U, s, V)` with
/// `U` `m×k`, `s` length `k`, `V` `n×k`.
pub fn svd_via_gram(a: &Mat) -> (Mat, Vec<f64>, Mat) {
    let (m, n) = a.shape();
    let k = m.min(n);

    if m >= n {
        // Eigen-decompose AᵀA (n×n): V = eigenvectors, σ = sqrt(λ), U = A V / σ.
        let gram = crate::ops::matmul::syrk_ata(a);
        let eig = SymEig::new(&gram);
        let v_full = eig.vectors();
        let s: Vec<f64> = eig.values().iter().map(|&l| l.max(0.0).sqrt()).collect();

        let v = v_full.cols_range(0, k);
        // U_t = A v_t / σ_t
        let av = matmul(a, &v); // m×k
        let mut u = av;
        let mut s_thin = vec![0.0; k];
        for t in 0..k {
            s_thin[t] = s[t];
            let col = &mut u.data[t * m..(t + 1) * m];
            let nrm = nrm2(col);
            if nrm > 0.0 {
                let inv = 1.0 / nrm;
                for x in col.iter_mut() {
                    *x *= inv;
                }
            }
        }
        (u, s_thin, v)
    } else {
        // Eigen-decompose A Aᵀ (m×m): U = eigenvectors, σ = sqrt(λ), V = Aᵀ U / σ.
        let gram = crate::ops::matmul::syrk_aat(a);
        let eig = SymEig::new(&gram);
        let u_full = eig.vectors();
        let s: Vec<f64> = eig.values().iter().map(|&l| l.max(0.0).sqrt()).collect();

        let u = u_full.cols_range(0, k);
        let atu = matmul(&a.transpose(), &u); // n×k
        let mut v = atu;
        let mut s_thin = vec![0.0; k];
        for t in 0..k {
            s_thin[t] = s[t];
            let col = &mut v.data[t * n..(t + 1) * n];
            let nrm = nrm2(col);
            if nrm > 0.0 {
                let inv = 1.0 / nrm;
                for x in col.iter_mut() {
                    *x *= inv;
                }
            }
        }
        (u, s_thin, v)
    }
}
