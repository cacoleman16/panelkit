//! Elementary orthogonal transforms shared by QR, SVD and the symmetric
//! eigensolver: Householder reflectors and Givens rotations.

use super::norms::nrm2;

/// A Householder reflector `H = I - beta * v vᵀ` with `v[0] == 1` (implicit).
///
/// Generated from a column `x` so that `H x = ±‖x‖ e_1`.
pub struct Householder {
    /// Reflector vector with the leading 1 stored explicitly in `v[0]`.
    pub v: Vec<f64>,
    pub beta: f64,
    /// The resulting first component, `∓‖x‖` (sign chosen for stability).
    pub alpha: f64,
}

impl Householder {
    /// Build the reflector that zeroes all but the first entry of `x`.
    ///
    /// Uses the LAPACK sign convention (reflect away from `x[0]`) to avoid
    /// cancellation. If `x` is already a multiple of `e_1`, `beta == 0`.
    pub fn new(x: &[f64]) -> Householder {
        let n = x.len();
        let mut v = x.to_vec();
        if n == 0 {
            return Householder {
                v,
                beta: 0.0,
                alpha: 0.0,
            };
        }
        let x0 = x[0];
        // norm of x[1..]
        let sigma = nrm2(&x[1..]);
        if sigma == 0.0 && x0 >= 0.0 {
            // Already aligned, no reflection.
            v[0] = 1.0;
            return Householder {
                v,
                beta: 0.0,
                alpha: x0,
            };
        }
        // hypot scales internally, so the norm neither overflows (‖x‖ ≳ 1e154
        // would overflow the naive square) nor flushes to zero (‖x‖ ≲ 1e-160).
        let xnorm = x0.hypot(sigma);
        // alpha = -sign(x0) * ||x||
        let alpha = if x0 <= 0.0 { xnorm } else { -xnorm };
        // v0 = x0 - alpha is nonzero here: the only zero case (sigma == 0 with
        // x0 >= 0) returned early above. The tail entries are O(1) after the
        // 1/v0 normalization, so beta = 2/vᵀv is scale-free.
        let v0 = x0 - alpha;
        let inv_v0 = 1.0 / v0;
        v[0] = 1.0;
        for vi in v.iter_mut().skip(1) {
            *vi *= inv_v0;
        }
        let mut vtv = 1.0;
        for &vi in v.iter().skip(1) {
            vtv += vi * vi;
        }
        let beta = 2.0 / vtv;
        Householder { v, beta, alpha }
    }

    /// Apply `H` from the left to a column-major block `a` with `rows` rows and
    /// `cols` columns, operating on the trailing sub-rows starting at `row0`.
    /// `a` is the full matrix buffer; `lead` is its row count (stride).
    pub fn apply_left(&self, a: &mut [f64], lead: usize, row0: usize, cols: usize) {
        if self.beta == 0.0 {
            return;
        }
        let vlen = self.v.len();
        for j in 0..cols {
            let base = j * lead + row0;
            // w = vᵀ a_col
            let mut w = 0.0;
            for k in 0..vlen {
                w += self.v[k] * a[base + k];
            }
            w *= self.beta;
            for k in 0..vlen {
                a[base + k] -= w * self.v[k];
            }
        }
    }
}

/// A Givens rotation `[[c, s], [-s, c]]` that zeroes the second component of
/// `(a, b)`: `[c s; -s c]ᵀ · (a, b) = (r, 0)`.
#[derive(Clone, Copy, Debug)]
pub struct Givens {
    pub c: f64,
    pub s: f64,
    pub r: f64,
}

impl Givens {
    /// Construct a stable Givens rotation from `(a, b)`.
    pub fn new(a: f64, b: f64) -> Givens {
        if b == 0.0 {
            Givens {
                c: if a >= 0.0 { 1.0 } else { -1.0 },
                s: 0.0,
                r: a.abs(),
            }
        } else if a == 0.0 {
            Givens {
                c: 0.0,
                s: if b >= 0.0 { 1.0 } else { -1.0 },
                r: b.abs(),
            }
        } else if a.abs() > b.abs() {
            let t = b / a;
            let u = (1.0 + t * t).sqrt().copysign(a);
            let c = 1.0 / u;
            Givens {
                c,
                s: c * t,
                r: a * u,
            }
        } else {
            let t = a / b;
            let u = (1.0 + t * t).sqrt().copysign(b);
            let s = 1.0 / u;
            Givens {
                c: s * t,
                s,
                r: b * u,
            }
        }
    }

    /// Apply to a pair `(x, y)`, returning the rotated pair.
    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (self.c * x + self.s * y, -self.s * x + self.c * y)
    }
}
