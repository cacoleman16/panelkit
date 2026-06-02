//! Fixed-effects "within" transforms (Frisch–Waugh–Lovell).
//!
//! For a **balanced** panel the two-way within transform has a closed form:
//! ```text
//!   ỹ_it = y_it − ȳ_i· − ȳ_·t + ȳ_··
//! ```
//! (subtract the unit mean and the time mean, add back the grand mean). This
//! absorbs unit and time fixed effects exactly. panelkit panels are balanced,
//! so we use the closed form; an iterative alternating-projections version is
//! provided for the (future) unbalanced case.

use panelkit_linalg::Mat;

/// Row (unit) means of an `N×T` matrix.
pub fn unit_means(m: &Mat) -> Vec<f64> {
    let (n, t) = m.shape();
    let mut means = vec![0.0; n];
    for j in 0..t {
        let col = m.col(j);
        for i in 0..n {
            means[i] += col[i];
        }
    }
    let inv = 1.0 / t as f64;
    means.iter_mut().for_each(|v| *v *= inv);
    means
}

/// Column (time) means of an `N×T` matrix.
pub fn time_means(m: &Mat) -> Vec<f64> {
    let (n, t) = m.shape();
    let mut means = vec![0.0; t];
    for j in 0..t {
        let col = m.col(j);
        means[j] = col.iter().sum::<f64>() / n as f64;
    }
    means
}

/// Grand mean of all entries.
pub fn grand_mean(m: &Mat) -> f64 {
    let s: f64 = m.as_slice().iter().sum();
    s / m.as_slice().len() as f64
}

/// Two-way within transform of a balanced panel (closed form).
pub fn two_way_within(m: &Mat) -> Mat {
    let (n, t) = m.shape();
    let um = unit_means(m);
    let tm = time_means(m);
    let gm = grand_mean(m);
    let mut out = Mat::zeros(n, t);
    for j in 0..t {
        for i in 0..n {
            out.set(i, j, m.get(i, j) - um[i] - tm[j] + gm);
        }
    }
    out
}

/// One-way (unit) within transform: subtract each unit's mean.
pub fn unit_within(m: &Mat) -> Mat {
    let (n, t) = m.shape();
    let um = unit_means(m);
    let mut out = Mat::zeros(n, t);
    for j in 0..t {
        for i in 0..n {
            out.set(i, j, m.get(i, j) - um[i]);
        }
    }
    out
}
