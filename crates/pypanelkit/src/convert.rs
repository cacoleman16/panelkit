//! The numpy ↔ `Mat` boundary.
//!
//! Discipline: **copy once** across the FFI boundary into the canonical
//! column-major `Mat` before any Rust-side work begins. numpy is the wire format
//! only; the core never links it. The copy (~200 KB at panelkit's sizes) also
//! avoids holding a GIL-borrowed buffer across any later parallel region.

use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use panelkit_linalg::Mat;
use pyo3::prelude::*;

/// Copy a (C- or F-contiguous) 2-D numpy array into a column-major [`Mat`].
#[allow(dead_code)]
pub fn mat_from_numpy(arr: &PyReadonlyArray2<f64>) -> Mat {
    let view = arr.as_array();
    let (rows, cols) = (view.shape()[0], view.shape()[1]);
    let mut m = Mat::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            m.set(i, j, view[[i, j]]);
        }
    }
    m
}

/// Copy a [`Mat`] out to a fresh row-major (C-order) numpy array.
#[allow(dead_code)]
pub fn mat_to_numpy<'py>(py: Python<'py>, m: &Mat) -> Bound<'py, PyArray2<f64>> {
    let (rows, cols) = m.shape();
    let row_major = m.to_row_major();
    // Build an ndarray view then convert; shape is (rows, cols), C-order.
    let arr = ndarray_from_row_major(rows, cols, row_major);
    arr.to_pyarray_bound(py)
}

#[allow(dead_code)]
fn ndarray_from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> numpy::ndarray::Array2<f64> {
    numpy::ndarray::Array2::from_shape_vec((rows, cols), data)
        .expect("row-major buffer matches shape")
}
