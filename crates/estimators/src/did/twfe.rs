//! Two-way fixed-effects difference-in-differences.
//!
//! Estimates `β` in `Y_it = α_i + γ_t + β·D_it + ε_it` via the two-way within
//! transform (FWL): demean both the outcome and the treatment indicator by unit
//! and time, then `β̂ = ⟨D̃, Ỹ⟩ / ⟨D̃, D̃⟩`. Standard errors are cluster-robust by
//! unit.
//!
//! Note: under staggered adoption with heterogeneous effects, β̂ is a
//! contaminated weighted average of treatment effects (Goodman-Bacon 2021) and
//! can even be sign-flipped. Use [`super::callaway`] or [`super::sunab`] for
//! staggered designs; the Goodman-Bacon decomposition in [`super::bacon`] is the
//! diagnostic for how bad the contamination is.

use crate::fe::within::two_way_within;
use crate::panel::Panel;
use panelkit_linalg::Mat;

/// Result of a TWFE fit.
#[derive(Clone, Debug)]
pub struct TwfeFit {
    /// TWFE coefficient on the treatment indicator.
    pub att: f64,
    /// Cluster-robust (by unit) standard error.
    pub se: f64,
    /// Residual degrees of freedom used in the SE.
    pub n_clusters: usize,
}

impl TwfeFit {
    /// t-statistic for `att = 0`.
    pub fn t_stat(&self) -> f64 {
        if self.se > 0.0 {
            self.att / self.se
        } else {
            f64::INFINITY
        }
    }
}

/// Treatment-indicator matrix `D_it` for a panel.
pub fn treatment_matrix(panel: &Panel) -> Mat {
    let (n, t) = (panel.n_units(), panel.n_periods());
    let mut d = Mat::zeros(n, t);
    for i in 0..n {
        for p in 0..t {
            if panel.is_treated(i, p) {
                d.set(i, p, 1.0);
            }
        }
    }
    d
}

/// Fit TWFE on a (possibly staggered) balanced panel.
pub fn fit(panel: &Panel) -> TwfeFit {
    let y = panel.y();
    let d = treatment_matrix(panel);

    let yt = two_way_within(y);
    let dt = two_way_within(&d);

    let (n, t) = (panel.n_units(), panel.n_periods());

    // β̂ = ⟨D̃, Ỹ⟩ / ⟨D̃, D̃⟩.
    let mut num = 0.0;
    let mut den = 0.0;
    for idx in 0..n * t {
        let dd = dt.as_slice()[idx];
        num += dd * yt.as_slice()[idx];
        den += dd * dd;
    }
    let beta = if den > 0.0 { num / den } else { 0.0 };

    // Cluster-robust SE by unit: V = (Σ_i g_i²) / den², g_i = Σ_t D̃_it ê_it.
    let mut meat = 0.0;
    for i in 0..n {
        let mut gi = 0.0;
        for p in 0..t {
            let dd = dt.get(i, p);
            let e = yt.get(i, p) - beta * dd;
            gi += dd * e;
        }
        meat += gi * gi;
    }
    // Small-sample cluster correction G/(G-1).
    let g = n as f64;
    let corr = if n > 1 { g / (g - 1.0) } else { 1.0 };
    let var = if den > 0.0 { corr * meat / (den * den) } else { 0.0 };

    TwfeFit {
        att: beta,
        se: var.sqrt(),
        n_clusters: n,
    }
}
