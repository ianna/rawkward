// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::PyObject;

use crate::content::{Content, ListOffsetArray, NumpyArray};
use super::layout::{PyIndex, PyNumpyArray, PyListOffsetArray};

pub fn from_python_nested_list(obj: &Bound<'_, PyAny>) -> PyResult<Content> {
    let outer: Vec<Vec<f64>> = obj.extract()?;

    let mut offsets = Vec::with_capacity(outer.len() + 1);
    offsets.push(0);

    for sub in &outer {
        let sub_len: i64 = sub.len() as i64;
        offsets.push(offsets.last().unwrap() + sub_len);
    }

    let flat: Vec<f64> = outer.into_iter().flatten().collect();
    let flat_len = flat.len();
    let data = Arc::from(flat.into_boxed_slice());

    Ok(Content::ListOffsetArray(ListOffsetArray {
        offsets: Arc::from(offsets.into_boxed_slice()),
        content: Arc::new(Content::NumpyArray(NumpyArray {
            data,
            shape: vec![flat_len],
            strides: vec![1],
        })),
    }))
}

pub fn content_to_python(py: Python, c: &Content) -> PyResult<PyObject> {
    match c {
        Content::NumpyArray(a) => {
            let obj = PyNumpyArray {
                data: a.data.to_vec(),
                shape: a.shape.clone(),
                strides: a.strides.clone(),
            };
            Py::new(py, obj).map(|p| p.into_bound(py).into_any().unbind())
        }

        Content::ListOffsetArray(a) => {
            let offsets = PyIndex { data: a.offsets.to_vec() };
            let content = content_to_python(py, &a.content)?;

            let obj = PyListOffsetArray {
                offsets,
                content,
            };
            Py::new(py, obj).map(|p| p.into_bound(py).into_any().unbind())
        }

        _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "layout conversion not implemented",
        )),
    }
}
