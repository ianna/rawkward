// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use rawkward::content::{Content, NumpyArray};
use rawkward::kernels::{Slice, slice};
use rawkward::dtype::DType;

#[pyclass]
pub struct PyArray {
    inner: Arc<Content>,
}

#[pymethods]
impl PyArray {
    #[new]
    fn new(obj: Bound<'_, PyAny>) -> PyResult<Self> {
        // Minimal: accept a Python list of floats
        let seq = obj.extract::<Vec<f64>>()?;
        let data: Arc<[f64]> = Arc::from(seq.into_boxed_slice());

        let array = NumpyArray { 
            data,
            shape: vec![obj.len()?],
            strides: vec![8],
            dtype: DType::F64,
        };

        Ok(PyArray {
            inner: Arc::new(Content::F64(array)),
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

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        let list = pyo3::types::PyList::new(py, vec![1.0, 2.0, 3.0])?;
        let py_array = PyArray::new(list.as_any().clone())?;
        
        println!("Array created! Length: {}", py_array.__len__());
        Ok(())
    })
}