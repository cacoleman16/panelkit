//! Dense, column-major `f64` matrix — the single storage type the whole core
//! is built on.
//!
//! Column-major (Fortran / BLAS convention) is deliberate: QR and SVD operate
//! column-by-column, so contiguous columns keep those hot loops cache-friendly.
//! Element `(i, j)` of an `r x c` matrix lives at `data[i + j * r]`.

use core::fmt;

/// A dense matrix of `f64`, stored column-major.
#[derive(Clone, PartialEq)]
pub struct Mat {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    /// Length `rows * cols`, column-major.
    pub(crate) data: Vec<f64>,
}

impl Mat {
    /// New `rows x cols` matrix filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// New matrix filled with a constant.
    pub fn filled(rows: usize, cols: usize, value: f64) -> Self {
        Mat {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    /// `n x n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Mat::zeros(n, n);
        for i in 0..n {
            m.data[i + i * n] = 1.0;
        }
        m
    }

    /// Wrap an existing column-major buffer. Panics if `data.len() != rows * cols`.
    pub fn from_col_major(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "from_col_major: data length {} != rows*cols {}",
            data.len(),
            rows * cols
        );
        Mat { rows, cols, data }
    }

    /// Build from a row-major slice (the common interchange layout, e.g. C-order
    /// numpy). Transposes into the canonical column-major storage.
    pub fn from_row_major(rows: usize, cols: usize, data: &[f64]) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "from_row_major: data length {} != rows*cols {}",
            data.len(),
            rows * cols
        );
        let mut m = Mat::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m.data[i + j * rows] = data[i * cols + j];
            }
        }
        m
    }

    /// Build a column vector (`n x 1`) from a slice.
    pub fn from_col_vec(data: &[f64]) -> Self {
        Mat {
            rows: data.len(),
            cols: 1,
            data: data.to_vec(),
        }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Read-only access to the underlying column-major buffer.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Mutable access to the underlying column-major buffer.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Export as a fresh row-major buffer (for numpy C-order output).
    pub fn to_row_major(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.rows * self.cols];
        for j in 0..self.cols {
            for i in 0..self.rows {
                out[i * self.cols + j] = self.data[i + j * self.rows];
            }
        }
        out
    }

    /// Element accessor. Bounds are `debug_assert!`-checked only (these are the
    /// hot-loop primitives): in release builds a row index past `rows` aliases
    /// into a neighboring column rather than panicking, so callers own the
    /// bounds. The slice index still panics when the *flat* offset is past the
    /// end of the buffer.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.rows && j < self.cols);
        self.data[i + j * self.rows]
    }

    /// See [`Mat::get`] for the (debug-only) bounds contract.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        debug_assert!(i < self.rows && j < self.cols);
        self.data[i + j * self.rows] = v;
    }

    #[inline]
    pub fn add_to(&mut self, i: usize, j: usize, v: f64) {
        debug_assert!(i < self.rows && j < self.cols);
        self.data[i + j * self.rows] += v;
    }

    /// Slice of column `j` (contiguous, length `rows`).
    #[inline]
    pub fn col(&self, j: usize) -> &[f64] {
        debug_assert!(j < self.cols);
        &self.data[j * self.rows..(j + 1) * self.rows]
    }

    /// Mutable slice of column `j`.
    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        debug_assert!(j < self.cols);
        &mut self.data[j * self.rows..(j + 1) * self.rows]
    }

    /// Copy row `i` into a fresh `Vec` (strided gather — avoid in hot loops).
    pub fn row_copy(&self, i: usize) -> Vec<f64> {
        debug_assert!(i < self.rows);
        (0..self.cols)
            .map(|j| self.data[i + j * self.rows])
            .collect()
    }

    /// Transpose into a new matrix.
    pub fn transpose(&self) -> Mat {
        let mut t = Mat::zeros(self.cols, self.rows);
        for j in 0..self.cols {
            for i in 0..self.rows {
                t.data[j + i * self.cols] = self.data[i + j * self.rows];
            }
        }
        t
    }

    /// Extract a submatrix from the given row/column index lists (gather).
    pub fn select(&self, row_idx: &[usize], col_idx: &[usize]) -> Mat {
        let mut out = Mat::zeros(row_idx.len(), col_idx.len());
        for (jo, &j) in col_idx.iter().enumerate() {
            for (io, &i) in row_idx.iter().enumerate() {
                out.set(io, jo, self.get(i, j));
            }
        }
        out
    }

    /// Select a contiguous range of columns `[start, end)`.
    pub fn cols_range(&self, start: usize, end: usize) -> Mat {
        debug_assert!(start <= end && end <= self.cols);
        Mat {
            rows: self.rows,
            cols: end - start,
            data: self.data[start * self.rows..end * self.rows].to_vec(),
        }
    }
}

impl fmt::Debug for Mat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Mat {}x{} [", self.rows, self.cols)?;
        for i in 0..self.rows {
            write!(f, "  ")?;
            for j in 0..self.cols {
                write!(f, "{:>12.6} ", self.get(i, j))?;
            }
            writeln!(f)?;
        }
        write!(f, "]")
    }
}
