//! Market selection: search candidate treatment-market sets and rank them by how
//! powerful, well-fitting, and trustworthy each design is.
//!
//! The combinatorial space of treated-market subsets is large, so we sample it
//! (plus always include every single market), score each candidate with a quick
//! power probe + diagnostics, and rank. Scoring each candidate is independent →
//! the search parallelizes across candidates.

use crate::diagnostics::diagnostics;
use crate::power::power_curve;
use crate::types::Method;
use panelkit_inference::par_map_items;
use panelkit_linalg::rng::Xoshiro256pp;
use panelkit_linalg::Mat;

/// A scored candidate treatment-market set.
#[derive(Clone, Debug)]
pub struct MarketCandidate {
    pub treated: Vec<usize>,
    /// Detection rate at the target lift.
    pub power_at_target: f64,
    /// Minimum detectable lift (fraction) at the requested power, if reached.
    pub mde_pct: Option<f64>,
    pub holdout_pct: f64,
    pub pre_fit_rel: f64,
    pub stability_score: f64,
    pub confidence: f64,
    /// Composite ranking score (higher is better).
    pub score: f64,
}

/// Configuration for a market search.
#[derive(Clone, Debug)]
pub struct SelectConfig {
    /// Units eligible to be treated (e.g. markets you could actually run in).
    pub eligible: Vec<usize>,
    /// Units **forced into every** candidate treatment set (must-treat markets).
    /// The search fills the remaining slots from `eligible`. Empty = no forcing.
    pub include: Vec<usize>,
    /// Maximum number of treated markets in a candidate set (counts `include`).
    pub max_treated: usize,
    pub test_len: usize,
    /// The lift you care about detecting (fraction, e.g. 0.05 = 5%).
    pub target_lift: f64,
    pub method: Method,
    pub alpha: f64,
    pub target_power: f64,
    pub min_pre: usize,
    /// How many candidate sets to sample/evaluate.
    pub n_candidates: usize,
    pub seed: u64,
    /// If `Some(k)`, only consider candidate sets of **exactly** `k` markets
    /// (used by the spec sweep so each "#geos" row reflects that size). If
    /// `None`, considers all sizes from 1 to `max_treated`.
    pub exact_size: Option<usize>,
    /// Number of most-recent historical placebo windows to power over.
    /// `None` = all available windows.
    pub lookback: Option<usize>,
}

/// Evaluate a single candidate set: quick power probe + diagnostics → score.
pub fn evaluate(y: &Mat, treated: &[usize], cfg: &SelectConfig) -> MarketCandidate {
    let tl = cfg.target_lift;
    let grid = [0.0, 0.5 * tl, tl, 1.5 * tl, 2.0 * tl];
    let pr = power_curve(
        y,
        treated,
        cfg.test_len,
        &grid,
        cfg.method,
        cfg.alpha,
        cfg.target_power,
        cfg.min_pre,
        cfg.lookback,
    );
    let power_at_target = pr
        .points
        .iter()
        .find(|p| (p.lift_pct - tl).abs() < 1e-12)
        .map(|p| p.power)
        .unwrap_or(0.0);
    let diag = diagnostics(y, treated, cfg.test_len);

    // Composite score: reward designs that both detect the target lift and are
    // trustworthy (high confidence). Small MDE breaks ties.
    let mde_term = pr.mde_pct.map(|m| 1.0 / (1.0 + m)).unwrap_or(0.0);
    let score = power_at_target * (0.5 + 0.5 * diag.confidence / 100.0) + 0.05 * mde_term;

    MarketCandidate {
        treated: treated.to_vec(),
        power_at_target,
        mde_pct: pr.mde_pct,
        holdout_pct: diag.holdout_pct,
        pre_fit_rel: diag.pre_fit_rel,
        stability_score: diag.stability_score,
        confidence: diag.confidence,
        score,
    }
}

/// Build the candidate list. Every candidate always contains the forced
/// `include` markets; the remaining slots are drawn from `eligible` (minus the
/// forced ones). With `exact_size = Some(k)`, every candidate has exactly `k`
/// markets total; otherwise it's the forced set plus each single extra market
/// plus sampled larger subsets up to `max_treated`.
fn candidate_sets(cfg: &SelectConfig) -> Vec<Vec<usize>> {
    let mut rng = Xoshiro256pp::seed_from_u64(cfg.seed);
    let mut seen: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();
    let mut sets: Vec<Vec<usize>> = Vec::new();

    // Forced (must-treat) markets, de-duplicated, and the pool of extra picks.
    // `eligible` is de-duplicated too: a repeated market must not produce
    // candidates that treat the same unit "twice" (which would double-count its
    // volume in the diagnostics).
    let mut forced: Vec<usize> = cfg.include.clone();
    forced.sort_unstable();
    forced.dedup();
    let forced_set: std::collections::HashSet<usize> = forced.iter().copied().collect();
    let mut extra_pool: Vec<usize> = cfg
        .eligible
        .iter()
        .copied()
        .filter(|u| !forced_set.contains(u))
        .collect();
    extra_pool.sort_unstable();
    extra_pool.dedup();

    if let Some(k0) = cfg.exact_size {
        let k = k0.max(1);
        if forced.len() > k {
            // Over-constrained: more must-treat markets than the requested set
            // size. No candidate can satisfy both — surface "no candidates"
            // rather than silently returning an oversized set.
            return sets;
        }
        let need = k - forced.len();
        if need == 0 {
            // The forced set already fills the requested size.
            if !forced.is_empty() {
                sets.push(forced.clone());
            }
            return sets;
        }
        if need == 1 {
            // Deterministic: forced + each eligible single (preserves the old
            // "all singletons" behavior when nothing is forced and k == 1).
            for &u in &extra_pool {
                let mut pick = forced.clone();
                pick.push(u);
                pick.sort_unstable();
                if seen.insert(pick.clone()) {
                    sets.push(pick);
                }
            }
            return sets;
        }
        let mut attempts = 0;
        while sets.len() < cfg.n_candidates && attempts < cfg.n_candidates * 40 {
            attempts += 1;
            let mut pool = extra_pool.clone();
            rng.shuffle(&mut pool);
            let mut pick: Vec<usize> = forced.clone();
            pick.extend(pool.into_iter().take(need));
            pick.sort_unstable();
            if pick.len() == k && seen.insert(pick.clone()) {
                sets.push(pick);
            }
        }
        return sets;
    }

    // Mixed-size search. Extra slots available on top of the forced set.
    let budget = cfg.max_treated.saturating_sub(forced.len());
    if !forced.is_empty() {
        seen.insert(forced.clone());
        sets.push(forced.clone());
    }
    if budget >= 1 {
        for &u in &extra_pool {
            let mut pick = forced.clone();
            pick.push(u);
            pick.sort_unstable();
            if seen.insert(pick.clone()) {
                sets.push(pick);
            }
        }
    }
    if budget >= 2 && extra_pool.len() >= 2 {
        let mut attempts = 0;
        while sets.len() < cfg.n_candidates && attempts < cfg.n_candidates * 20 {
            attempts += 1;
            let extra = 2 + rng.gen_range(budget - 1); // 2..=budget extra markets
            let mut pool = extra_pool.clone();
            rng.shuffle(&mut pool);
            let mut pick: Vec<usize> = forced.clone();
            pick.extend(pool.into_iter().take(extra));
            pick.sort_unstable();
            if seen.insert(pick.clone()) {
                sets.push(pick);
            }
        }
    }
    sets
}

/// Search and rank candidate treatment-market sets (descending by score).
pub fn select_markets(y: &Mat, cfg: &SelectConfig) -> Vec<MarketCandidate> {
    let candidates = candidate_sets(cfg);
    let mut scored: Vec<MarketCandidate> =
        par_map_items(candidates, |treated| evaluate(y, &treated, cfg));
    // total_cmp: a NaN score (degenerate panel) must rank last, not panic.
    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    scored
}
