//! Constrained-optimization solvers used by the estimators: simplex-constrained
//! QP (synthetic-control weights) and the SVT prox operator (MC-NNM).

pub mod simplex;
pub mod softthresh;

pub use simplex::{
    project_bounded_simplex, project_simplex, sc_weights, sc_weights_bounded, solve_bounded,
    solve_fw, solve_pg, SimplexSolution, WeightBounds,
};
pub use softthresh::{svt, svt_from, svt_truncated};
