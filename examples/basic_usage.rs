// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::content::{Content, NumpyArray};
use crate::kernels::{slice, Slice};

#[pyclass]
pub struct PyArray {
    inner: Arc<Content>,
}

#[pymethods]
impl PyArray {
    #[new]
    fn new(obj: &PyAny) -> PyResult<Self> {
        // Minimal: accept a Python list of floats
        let seq = obj.extract::<Vec<f64>>()?;
        let data: Arc<[f64]> = Arc::from(seq.into_boxed_slice());
        Ok(PyArray {
            inner: Arc::new(Content::NumpyArray(NumpyArray { data })),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, idx: isize) -> PyResult<PyArray> {
        let s = Slice::Index(idx as i64);
        let out = slice(&self.inner, &s)
            .map_err(|e| pyo3::exceptions::PyIndexError::new_err(e.to_string()))?;
        Ok(PyArray {
            inner: Arc::new(out),
        })
    }
}
