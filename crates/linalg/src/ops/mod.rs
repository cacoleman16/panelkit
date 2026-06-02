//! Level-1/2/3 building blocks: matmul kernels, norms, and the orthogonal
//! transforms used by the factorizations.

pub mod matmul;
pub mod norms;
pub mod transform;

pub use matmul::{gemm, gemv, gemv_t, matmul, matvec, matvec_t, syrk_aat, syrk_ata};
pub use norms::{axpy, dot, frobenius, nrm2, nrm_inf, sum_sq};
pub use transform::{Givens, Householder};
