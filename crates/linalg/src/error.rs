//! Error type for the numerical core.

use core::fmt;

/// Errors raised by factorizations and solvers.
#[derive(Clone, Debug, PartialEq)]
pub enum LinalgError {
    /// A matrix expected to be square was not.
    NotSquare { rows: usize, cols: usize },
    /// Dimensions of two operands were incompatible.
    DimMismatch { expected: usize, got: usize },
    /// A matrix expected to be symmetric-positive-definite had a non-positive
    /// pivot during Cholesky (i.e. it is indefinite or singular).
    NotPositiveDefinite { pivot: usize },
    /// An iterative solver failed to reach tolerance within the iteration cap.
    DidNotConverge { iters: usize },
    /// The problem was empty / ill-posed (e.g. zero columns).
    Empty,
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinalgError::NotSquare { rows, cols } => {
                write!(f, "matrix is not square ({rows}x{cols})")
            }
            LinalgError::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            LinalgError::NotPositiveDefinite { pivot } => {
                write!(f, "matrix is not positive definite (pivot {pivot} <= 0)")
            }
            LinalgError::DidNotConverge { iters } => {
                write!(f, "iterative method did not converge in {iters} iterations")
            }
            LinalgError::Empty => write!(f, "empty or ill-posed problem"),
        }
    }
}

impl std::error::Error for LinalgError {}

pub type Result<T> = core::result::Result<T, LinalgError>;
