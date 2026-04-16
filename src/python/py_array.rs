// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#![cfg(feature = "python")]

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

use crate::content::{Content, NumpyArray};
use crate::kernels::{slice, Slice};
use crate::python::convert::from_python_nested_list;
use crate::python::convert::content_to_python;


#[pyclass(name = "Array", module = "rawkward")]
pub struct PyArray {
    pub(crate) inner: Arc<Content>,
}

#[pymethods]
impl PyArray {
    #[new]
    fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<PyList>() {
            let first = obj.get_item(0)?;
            if first.is_instance_of::<PyList>() {
                let content = from_python_nested_list(obj)?;
                return Ok(PyArray { inner: Arc::new(content) });
            }
        }

        let seq = obj.extract::<Vec<f64>>()?;
        let len = seq.len();
        let data = Arc::from(seq.into_boxed_slice());
        Ok(PyArray {
            inner: Arc::new(Content::NumpyArray(NumpyArray {
                data,
                shape: vec![len],
                strides: vec![1],
            })),
        })
    }
    
    #[getter]
    pub fn layout(&self, py: Python) -> PyResult<PyObject> {
        content_to_python(py, &self.inner)
    }
    
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, idx: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        // Case 1: integer index
        if let Ok(i) = idx.extract::<isize>() {
            let s = Slice::Index(i as i64);
            let out = slice(&self.inner, &s)
                .map_err(|e: crate::kernels::SliceError| pyo3::exceptions::PyIndexError::new_err(e.to_string()))?; //.map_err(|e| pyo3::exceptions::PyIndexError::new_err(e.to_string()))?;
            return Ok(PyArray { inner: Arc::new(out) });
        }
        // Case 2: Python slice object
        if let Ok(py_slice) = idx.downcast::<pyo3::types::PySlice>() {
            let indices = py_slice.indices((self.inner.len() as i64).try_into().unwrap())?;
            let start = indices.start as usize;
            let stop = indices.stop as usize;

            if indices.step != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err("slice step must be 1"));
            }

            // Delegate to a range slice on the content directly
            let out = crate::kernels::slice_range(&self.inner, start, stop)
                .map_err(|e: crate::kernels::SliceError| pyo3::exceptions::PyIndexError::new_err(e.to_string()))?; // .map_err(|e| pyo3::exceptions::PyIndexError::new_err(e.to_string()))?;
            return Ok(PyArray { inner: Arc::new(out) });
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "index must be int or slice",
        ))
    }

    fn __repr__(&self) -> PyResult<String> {
        fn fmt(c: &Content) -> String {
            match c {
                Content::NumpyArray(a) => format!("Array({:?})", &*a.data),
                Content::ListOffsetArray(a) => {
                    let items: Vec<String> = (0..a.len())
                        .map(|i| a.slice(i).map(|c| fmt(&c)).unwrap_or("?".into()))
                        .collect();
                    format!("Array([{}])", items.join(", "))
                }
                _ => format!("Array(...)"),
            }
        }
        Ok(fmt(&self.inner))
    }
}
