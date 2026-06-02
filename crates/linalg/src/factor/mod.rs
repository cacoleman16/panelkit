//! Matrix factorizations: Cholesky, Householder QR, one-sided Jacobi SVD,
//! symmetric (cyclic Jacobi) eigendecomposition, plus a Gram-based SVD oracle.

pub mod cholesky;
pub mod eig_sym;
pub mod qr;
pub mod svd;
pub mod svd_gram;

pub use cholesky::Cholesky;
pub use eig_sym::SymEig;
pub use qr::Qr;
pub use svd::Svd;
