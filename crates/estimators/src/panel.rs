//! The panel data container shared by every estimator.
//!
//! Outcomes are stored as an `N×T` [`Mat`] (rows = units, columns = time
//! periods). Treatment timing is encoded per unit as the period index at which
//! the unit first becomes treated (`None` = never treated). This representation
//! covers both the block-treatment case the SC family expects (all treated units
//! share one start period) and the staggered-adoption case the DiD family needs.

use panelkit_linalg::Mat;

/// A balanced panel of outcomes plus treatment timing.
#[derive(Clone)]
pub struct Panel {
    /// Outcomes, `N×T` (rows = units, cols = periods).
    y: Mat,
    /// Per-unit first-treated period; `None` for never-treated units.
    treat_start: Vec<Option<usize>>,
}

impl Panel {
    /// Construct from an outcome matrix and per-unit treatment-start periods.
    /// Panics if `treat_start.len() != y.rows()`.
    pub fn new(y: Mat, treat_start: Vec<Option<usize>>) -> Panel {
        assert_eq!(
            treat_start.len(),
            y.rows(),
            "treat_start length {} != n_units {}",
            treat_start.len(),
            y.rows()
        );
        Panel { y, treat_start }
    }

    /// Construct a block-treatment panel: the units in `treated` all begin
    /// treatment at `treat_time`; all other units are controls.
    pub fn block(y: Mat, treated: &[usize], treat_time: usize) -> Panel {
        let n = y.rows();
        let mut starts = vec![None; n];
        for &u in treated {
            assert!(u < n, "treated unit index {u} out of range");
            starts[u] = Some(treat_time);
        }
        Panel::new(y, starts)
    }

    #[inline]
    pub fn n_units(&self) -> usize {
        self.y.rows()
    }

    #[inline]
    pub fn n_periods(&self) -> usize {
        self.y.cols()
    }

    /// The outcome matrix (`N×T`).
    #[inline]
    pub fn y(&self) -> &Mat {
        &self.y
    }

    /// Outcome for `(unit, period)`.
    #[inline]
    pub fn outcome(&self, unit: usize, period: usize) -> f64 {
        self.y.get(unit, period)
    }

    /// Per-unit treatment-start periods.
    #[inline]
    pub fn treat_start(&self) -> &[Option<usize>] {
        &self.treat_start
    }

    /// Whether `(unit, period)` is under treatment.
    #[inline]
    pub fn is_treated(&self, unit: usize, period: usize) -> bool {
        matches!(self.treat_start[unit], Some(t0) if period >= t0)
    }

    /// Indices of ever-treated units.
    pub fn treated_units(&self) -> Vec<usize> {
        (0..self.n_units())
            .filter(|&u| self.treat_start[u].is_some())
            .collect()
    }

    /// Indices of never-treated units (the clean control / donor pool).
    pub fn never_treated_units(&self) -> Vec<usize> {
        (0..self.n_units())
            .filter(|&u| self.treat_start[u].is_none())
            .collect()
    }

    /// The set of distinct treatment-start periods among treated units
    /// (the adoption "cohorts"), sorted ascending.
    pub fn cohorts(&self) -> Vec<usize> {
        let mut cs: Vec<usize> = self
            .treat_start
            .iter()
            .filter_map(|&t| t)
            .collect();
        cs.sort_unstable();
        cs.dedup();
        cs
    }

    /// For a common-treatment-time design, the single shared start period.
    /// Returns `None` if treated units do not all share one start period.
    pub fn common_treat_time(&self) -> Option<usize> {
        let cs = self.cohorts();
        match cs.len() {
            1 => Some(cs[0]),
            _ => None,
        }
    }

    /// Donor (never-treated) pre-period block: `T_pre × J` matrix whose columns
    /// are donor units' pre-treatment outcomes. Returns `(matrix, donor_ids)`.
    pub fn donor_pre(&self, treat_time: usize) -> (Mat, Vec<usize>) {
        let donors = self.never_treated_units();
        let mut m = Mat::zeros(treat_time, donors.len());
        for (jc, &u) in donors.iter().enumerate() {
            for t in 0..treat_time {
                m.set(t, jc, self.y.get(u, t));
            }
        }
        (m, donors)
    }

    /// Donor (never-treated) post-period block: `T_post × J`.
    pub fn donor_post(&self, treat_time: usize) -> (Mat, Vec<usize>) {
        let donors = self.never_treated_units();
        let t_post = self.n_periods() - treat_time;
        let mut m = Mat::zeros(t_post, donors.len());
        for (jc, &u) in donors.iter().enumerate() {
            for t in treat_time..self.n_periods() {
                m.set(t - treat_time, jc, self.y.get(u, t));
            }
        }
        (m, donors)
    }

    /// Average outcome across the given units at each period (length `T`).
    pub fn unit_mean(&self, units: &[usize]) -> Vec<f64> {
        let t = self.n_periods();
        let mut out = vec![0.0; t];
        if units.is_empty() {
            return out;
        }
        for &u in units {
            for p in 0..t {
                out[p] += self.y.get(u, p);
            }
        }
        let inv = 1.0 / units.len() as f64;
        out.iter_mut().for_each(|v| *v *= inv);
        out
    }
}
