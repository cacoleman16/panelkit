//! Goodman-Bacon (2021) decomposition of the two-way fixed-effects DiD estimate.
//!
//! Under staggered adoption the TWFE coefficient is an identity-weighted average
//! of *all* 2×2 difference-in-differences comparisons between pairs of timing
//! groups:
//!
//! ```text
//!   β^DD = Σ_k s_{kU} β_{kU}            (treated cohort vs never-treated)
//!        + Σ_{k<l} s_{kl}^k β_{kl}^k    (earlier cohort treated, later as clean control)
//!        + Σ_{k<l} s_{kl}^l β_{kl}^l    (LATER cohort treated, earlier as control — the
//!                                        "forbidden" comparison using already-treated units)
//! ```
//!
//! The forbidden comparisons are what bias TWFE under heterogeneous effects:
//! they subtract the *already-treated* earlier cohort's (possibly still-evolving)
//! outcome path. This module reports every component, its weight, and the total
//! weight resting on forbidden comparisons. As a built-in check,
//! `Σ weight · estimate` reproduces the TWFE coefficient.
//!
//! Assumes a balanced panel, sharp timing, and a never-treated group. Units
//! treated from period 0 (no pre-period) are excluded from the decomposition.

use crate::panel::Panel;

/// Which kind of 2×2 comparison a component represents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaconKind {
    /// Treated cohort vs never-treated.
    TreatedVsUntreated,
    /// Earlier cohort treated, later cohort as a (clean, not-yet-treated) control.
    EarlierVsLater,
    /// Later cohort treated, earlier cohort as control — uses already-treated
    /// units as controls (the bias-inducing "forbidden" comparison).
    LaterVsEarlierForbidden,
}

/// One 2×2 comparison in the decomposition.
#[derive(Clone, Debug)]
pub struct BaconComponent {
    pub kind: BaconKind,
    pub treated_cohort: usize,
    /// Comparison cohort; `None` means the never-treated group.
    pub comparison_cohort: Option<usize>,
    /// Normalized weight (weights over all components sum to 1).
    pub weight: f64,
    /// The 2×2 DiD estimate for this comparison.
    pub estimate: f64,
}

/// Result of the Goodman-Bacon decomposition.
#[derive(Clone, Debug)]
pub struct BaconResult {
    pub components: Vec<BaconComponent>,
    /// Weighted-average estimate `Σ wᵢ βᵢ` — equals the TWFE coefficient.
    pub twfe: f64,
    /// Total weight on forbidden (later-vs-earlier) comparisons.
    pub forbidden_weight: f64,
}

/// Mean outcome over `units` and periods `[lo, hi)`.
fn window_mean(panel: &Panel, units: &[usize], lo: usize, hi: usize) -> f64 {
    if units.is_empty() || hi <= lo {
        return 0.0;
    }
    let mut s = 0.0;
    for &u in units {
        for p in lo..hi {
            s += panel.outcome(u, p);
        }
    }
    s / (units.len() * (hi - lo)) as f64
}

/// Compute the Goodman-Bacon decomposition for a staggered panel.
pub fn decompose(panel: &Panel) -> BaconResult {
    let t = panel.n_periods() as f64;
    let never = panel.never_treated_units();
    let cohorts: Vec<usize> = panel.cohorts().into_iter().filter(|&g| g >= 1).collect();

    let units_of = |g: usize| -> Vec<usize> {
        (0..panel.n_units())
            .filter(|&i| panel.treat_start()[i] == Some(g))
            .collect()
    };

    // Total units entering the decomposition (treated cohorts + never-treated).
    let n_total: f64 =
        (cohorts.iter().map(|&g| units_of(g).len()).sum::<usize>() + never.len()) as f64;
    let share = |count: usize| count as f64 / n_total;

    // Fraction of periods treated for a cohort starting at g: periods [g, T).
    let dbar = |g: usize| (panel.n_periods() - g) as f64 / t;

    let n_u = share(never.len());

    let mut raw: Vec<(BaconKind, usize, Option<usize>, f64, f64)> = Vec::new();

    // --- Treated cohort vs never-treated. ---
    if !never.is_empty() {
        for &k in &cohorts {
            let uk = units_of(k);
            let nk = share(uk.len());
            let dk = dbar(k);
            let n_ku = nk / (nk + n_u);
            let s = (nk + n_u).powi(2) * n_ku * (1.0 - n_ku) * dk * (1.0 - dk);

            let yk_pre = window_mean(panel, &uk, 0, k);
            let yk_post = window_mean(panel, &uk, k, panel.n_periods());
            let yu_pre = window_mean(panel, &never, 0, k);
            let yu_post = window_mean(panel, &never, k, panel.n_periods());
            let beta = (yk_post - yk_pre) - (yu_post - yu_pre);

            raw.push((BaconKind::TreatedVsUntreated, k, None, s, beta));
        }
    }

    // --- Pairs of timing groups (k earlier, l later). ---
    for (ai, &k) in cohorts.iter().enumerate() {
        for &l in cohorts.iter().skip(ai + 1) {
            // cohorts is sorted ascending, so k < l (k treated earlier).
            let uk = units_of(k);
            let ul = units_of(l);
            let nk = share(uk.len());
            let nl = share(ul.len());
            let dk = dbar(k);
            let dl = dbar(l);
            let n_kl = nk / (nk + nl);

            // Earlier treated (k), later (l) as clean control: periods [0, l).
            {
                let yk_pre = window_mean(panel, &uk, 0, k);
                let yk_post = window_mean(panel, &uk, k, l);
                let yl_pre = window_mean(panel, &ul, 0, k);
                let yl_post = window_mean(panel, &ul, k, l);
                let beta = (yk_post - yk_pre) - (yl_post - yl_pre);
                let s = ((nk + nl) * (1.0 - dl)).powi(2)
                    * n_kl
                    * (1.0 - n_kl)
                    * ((dk - dl) / (1.0 - dl))
                    * ((1.0 - dk) / (1.0 - dl));
                raw.push((BaconKind::EarlierVsLater, k, Some(l), s, beta));
            }

            // Later treated (l), earlier (k) as ALREADY-TREATED control: periods [k, T).
            {
                let yl_pre = window_mean(panel, &ul, k, l);
                let yl_post = window_mean(panel, &ul, l, panel.n_periods());
                let yk_pre = window_mean(panel, &uk, k, l);
                let yk_post = window_mean(panel, &uk, l, panel.n_periods());
                let beta = (yl_post - yl_pre) - (yk_post - yk_pre);
                let s =
                    ((nk + nl) * dk).powi(2) * n_kl * (1.0 - n_kl) * (dl / dk) * ((dk - dl) / dk);
                raw.push((BaconKind::LaterVsEarlierForbidden, l, Some(k), s, beta));
            }
        }
    }

    // Normalize weights to sum to 1.
    let total_w: f64 = raw.iter().map(|r| r.3).sum();
    let mut components = Vec::with_capacity(raw.len());
    let mut twfe = 0.0;
    let mut forbidden_weight = 0.0;
    for (kind, treated, comp, s, beta) in raw {
        let w = if total_w > 0.0 { s / total_w } else { 0.0 };
        twfe += w * beta;
        if kind == BaconKind::LaterVsEarlierForbidden {
            forbidden_weight += w;
        }
        components.push(BaconComponent {
            kind,
            treated_cohort: treated,
            comparison_cohort: comp,
            weight: w,
            estimate: beta,
        });
    }

    BaconResult {
        components,
        twfe,
        forbidden_weight,
    }
}
