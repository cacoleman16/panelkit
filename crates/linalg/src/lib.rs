//! `panelkit-linalg` — a small, dependency-free, from-scratch dense linear
//! algebra and optimization core.
//!
//! Scope is deliberately narrow: everything panelkit's causal estimators need
//! and nothing else. All arithmetic is `f64`. Storage is column-major
//! ([`Mat`]). There are **no external numeric dependencies** (no BLAS/LAPACK/
//! ndarray/rand) — the crate is intended to be reusable as the numerical
//! foundation of sibling projects (e.g. a future time-series library).
//!
//! Module map:
//! - [`matrix`]  — the `Mat` storage type.
//! - [`ops`]     — matmul kernels, norms, Householder/Givens transforms.
//! - [`factor`]  — Cholesky, QR, SVD (Jacobi), symmetric eig.
//! - [`solve`]   — SPD and least-squares linear solves (+ ridge).
//! - [`opt`]     — simplex-constrained solvers and the SVT prox operator.
//! - [`rng`]     — deterministic, splittable PRNG.

// In dense linear algebra, index-based loops over `0..n` that touch several
// arrays at the same offset are clearer (and map more directly to the math)
// than zipped iterators. We opt out of this lint crate-wide deliberately.
#![allow(clippy::needless_range_loop)]

pub mod error;
pub mod factor;
pub mod matrix;
pub mod ops;
pub mod opt;
pub mod rng;
pub mod solve;

pub use error::{LinalgError, Result};
pub use matrix::Mat;

/// Common imports for downstream crates.
pub mod prelude {
    pub use crate::error::{LinalgError, Result};
    pub use crate::factor::{cholesky::Cholesky, qr::Qr, svd::Svd};
    pub use crate::matrix::Mat;
    pub use crate::ops::{matmul, matvec, matvec_t, syrk_ata};
    pub use crate::rng::Xoshiro256pp;
}

#[cfg(test)]
mod tests {
    use super::matrix::Mat;
    use super::ops::{matmul, syrk_ata};

    #[test]
    fn identity_roundtrip() {
        let i = Mat::identity(3);
        let a = Mat::from_row_major(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = matmul(&i, &a);
        assert_eq!(a, b);
    }

    #[test]
    fn row_major_col_major_roundtrip() {
        let a = Mat::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(a.get(0, 0), 1.0);
        assert_eq!(a.get(0, 2), 3.0);
        assert_eq!(a.get(1, 0), 4.0);
        assert_eq!(a.to_row_major(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn gram_is_symmetric() {
        let a = Mat::from_row_major(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let g = syrk_ata(&a);
        assert_eq!(g.shape(), (2, 2));
        // [1 3 5; 2 4 6] · [1 2; 3 4; 5 6]
        assert_eq!(g.get(0, 0), 1.0 + 9.0 + 25.0);
        assert_eq!(g.get(0, 1), 2.0 + 12.0 + 30.0);
        assert_eq!(g.get(1, 0), g.get(0, 1));
    }
}
