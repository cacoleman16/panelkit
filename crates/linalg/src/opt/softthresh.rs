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
