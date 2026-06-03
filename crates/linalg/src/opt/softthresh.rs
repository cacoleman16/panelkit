//! The singular-value-thresholding (SVT) prox operator of the nuclear norm —
//! the inner step of MC-NNM / SoftImpute.
//!
//! `SVT_λ(A) = U diag(max(σ − λ, 0)) Vᵀ`, i.e. soft-threshold the singular
//! values. Built directly on the production [`Svd`].

use crate::factor::svd::Svd;
use crate::matrix::Mat;

/// Apply singular-value soft-thresholding with threshold `lambda`. Returns the
/// thresholded matrix and the resulting nuclear norm (Σ of thresholded values).
pub fn svt(a: &Mat, lambda: f64) -> (Mat, f64) {
    let svd = Svd::new(a);
    let thresh: Vec<f64> = svd
        .singular_values()
        .iter()
        .map(|&s| (s - lambda).max(0.0))
        .collect();
    let nuc: f64 = thresh.iter().sum();
    (svd.reconstruct_with(&thresh), nuc)
}

/// SVT reusing a precomputed SVD (avoids recomputation when the caller already
/// has one).
pub fn svt_from(svd: &Svd, lambda: f64) -> (Mat, f64) {
    let thresh: Vec<f64> = svd
        .singular_values()
        .iter()
        .map(|&s| (s - lambda).max(0.0))
        .collect();
    let nuc: f64 = thresh.iter().sum();
    (svd.reconstruct_with(&thresh), nuc)
}

/// Truncated SVT via a randomized rank-`max_rank` SVD — much faster than a full
/// SVD when the result is low-rank (the usual MC-NNM case). Self-contained
/// (reuses panelkit's QR + Jacobi SVD), no LAPACK.
pub fn svt_truncated(a: &Mat, lambda: f64, max_rank: usize, seed: u64) -> (Mat, f64) {
    let rsvd = crate::factor::randomized::randomized_svd(a, max_rank, 6, 1, seed);
    let thresh: Vec<f64> = rsvd.s.iter().map(|&s| (s - lambda).max(0.0)).collect();
    let nuc: f64 = thresh.iter().sum();
    (rsvd.reconstruct_with(&thresh), nuc)
}
