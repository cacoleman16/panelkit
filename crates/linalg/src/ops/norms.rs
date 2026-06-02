//! Vector and matrix norms.

use crate::matrix::Mat;

/// Euclidean (2-) norm of a slice, computed with scaling to avoid overflow.
pub fn nrm2(x: &[f64]) -> f64 {
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for &xi in x {
        if xi != 0.0 {
            let a = xi.abs();
            if scale < a {
                let r = scale / a;
                ssq = 1.0 + ssq * r * r;
                scale = a;
            } else {
                let r = a / scale;
                ssq += r * r;
            }
        }
    }
    scale * ssq.sqrt()
}

/// Dot product of two equal-length slices.
#[inline]
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let mut acc = 0.0;
    for i in 0..x.len() {
        acc += x[i] * y[i];
    }
    acc
}

/// `y := y + alpha * x` (axpy).
#[inline]
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    for i in 0..x.len() {
        y[i] += alpha * x[i];
    }
}

/// Frobenius norm of a matrix.
pub fn frobenius(a: &Mat) -> f64 {
    nrm2(a.as_slice())
}

/// Sum of squares of a slice.
pub fn sum_sq(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

/// Max-absolute (infinity) norm of a slice.
pub fn nrm_inf(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
}
