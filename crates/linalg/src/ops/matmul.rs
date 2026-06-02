//! Dense matrix-multiply kernels, column-major.
//!
//! The kernels are written in *gaxpy* (generalized A·x-plus-y) form: because
//! storage is column-major, accumulating `C[:, j] += A[:, p] * B[p, j]` walks
//! contiguous columns of both `A` and `C`, which is cache-friendly without any
//! explicit tiling. At panelkit's matrix sizes (≈200×130) this is well within
//! "fast enough inside a multi-thousand-replicate bootstrap"; we deliberately do
//! not chase hand-tuned-BLAS throughput.

use crate::matrix::Mat;

/// `C := alpha * A * B + beta * C`.
///
/// `A` is `m×k`, `B` is `k×n`, `C` is `m×n`. Panics on shape mismatch.
pub fn gemm(alpha: f64, a: &Mat, b: &Mat, beta: f64, c: &mut Mat) {
    assert_eq!(a.cols(), b.rows(), "gemm: A.cols != B.rows");
    assert_eq!(a.rows(), c.rows(), "gemm: A.rows != C.rows");
    assert_eq!(b.cols(), c.cols(), "gemm: B.cols != C.cols");

    let m = a.rows();
    let k = a.cols();
    let n = b.cols();

    for j in 0..n {
        // Scale the destination column by beta first.
        let cj = c.col_mut(j);
        if beta == 0.0 {
            cj.iter_mut().for_each(|v| *v = 0.0);
        } else if beta != 1.0 {
            cj.iter_mut().for_each(|v| *v *= beta);
        }
    }

    if alpha == 0.0 {
        return;
    }

    for j in 0..n {
        for p in 0..k {
            // b[p, j]
            let bpj = alpha * b.get(p, j);
            if bpj == 0.0 {
                continue;
            }
            let acol = &a.data[p * m..(p + 1) * m];
            let ccol = &mut c.data[j * m..(j + 1) * m];
            for i in 0..m {
                ccol[i] += bpj * acol[i];
            }
        }
    }
}

/// Convenience: returns a freshly-allocated `A * B`.
pub fn matmul(a: &Mat, b: &Mat) -> Mat {
    let mut c = Mat::zeros(a.rows(), b.cols());
    gemm(1.0, a, b, 0.0, &mut c);
    c
}

/// `y := alpha * A * x + beta * y`, where `x` and `y` are length-compatible slices.
///
/// `A` is `m×n`, `x` has length `n`, `y` has length `m`.
pub fn gemv(alpha: f64, a: &Mat, x: &[f64], beta: f64, y: &mut [f64]) {
    let m = a.rows();
    let n = a.cols();
    assert_eq!(x.len(), n, "gemv: x length != A.cols");
    assert_eq!(y.len(), m, "gemv: y length != A.rows");

    if beta == 0.0 {
        y.iter_mut().for_each(|v| *v = 0.0);
    } else if beta != 1.0 {
        y.iter_mut().for_each(|v| *v *= beta);
    }
    if alpha == 0.0 {
        return;
    }
    for j in 0..n {
        let axj = alpha * x[j];
        if axj == 0.0 {
            continue;
        }
        let acol = &a.data[j * m..(j + 1) * m];
        for i in 0..m {
            y[i] += axj * acol[i];
        }
    }
}

/// Returns `A * x` as a fresh vector.
pub fn matvec(a: &Mat, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.rows()];
    gemv(1.0, a, x, 0.0, &mut y);
    y
}

/// `y := alpha * Aᵀ * x + beta * y`.
///
/// `A` is `m×n`, `x` has length `m`, `y` has length `n`. Computes each output as
/// a dot product over a contiguous column of `A`.
pub fn gemv_t(alpha: f64, a: &Mat, x: &[f64], beta: f64, y: &mut [f64]) {
    let m = a.rows();
    let n = a.cols();
    assert_eq!(x.len(), m, "gemv_t: x length != A.rows");
    assert_eq!(y.len(), n, "gemv_t: y length != A.cols");

    for j in 0..n {
        let acol = &a.data[j * m..(j + 1) * m];
        let mut acc = 0.0;
        for i in 0..m {
            acc += acol[i] * x[i];
        }
        if beta == 0.0 {
            y[j] = alpha * acc;
        } else {
            y[j] = beta * y[j] + alpha * acc;
        }
    }
}

/// Returns `Aᵀ * x` as a fresh vector.
pub fn matvec_t(a: &Mat, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.cols()];
    gemv_t(1.0, a, x, 0.0, &mut y);
    y
}

/// Symmetric rank-k product: returns `Aᵀ * A` (the `n×n` Gram matrix) for an
/// `m×n` matrix `A`. Exploits symmetry — only the upper triangle is computed
/// then mirrored — so it costs roughly half a full `gemm`.
pub fn syrk_ata(a: &Mat) -> Mat {
    let m = a.rows();
    let n = a.cols();
    let mut g = Mat::zeros(n, n);
    for j in 0..n {
        let aj = &a.data[j * m..(j + 1) * m];
        for i in 0..=j {
            let ai = &a.data[i * m..(i + 1) * m];
            let mut acc = 0.0;
            for t in 0..m {
                acc += ai[t] * aj[t];
            }
            g.set(i, j, acc);
            g.set(j, i, acc);
        }
    }
    g
}

/// Returns `A * Aᵀ` (the `m×m` Gram matrix) for an `m×n` matrix `A`.
pub fn syrk_aat(a: &Mat) -> Mat {
    // (A Aᵀ) = (Aᵀ)ᵀ (Aᵀ); reuse syrk_ata on the transpose.
    syrk_ata(&a.transpose())
}
