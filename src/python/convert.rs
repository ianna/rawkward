// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::PyObject;
use pyo3::types::{PyAny, PyDict, PyList};

use crate::content::{Content, ListOffsetArray, NumpyArray};
use super::layout::{PyIndex, PyNumpyArray, PyListOffsetArray};

pub fn from_python_object(obj: &Bound<'_, PyAny>) -> PyResult<Content> {
    // Try flat list of numbers first
    if let Ok(flat) = obj.extract::<Vec<f64>>() {
        let len = flat.len();
        let data = Arc::from(flat.into_boxed_slice());
        return Ok(Content::NumpyArray(NumpyArray {
            data,
            shape: vec![len],
            strides: vec![1],
        }));
    }

    // Must be a list of items — recurse
    let outer = obj.downcast::<PyList>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("expected a list"))?;

    let mut offsets: Vec<i64> = vec![0];
    let mut children: Vec<Content> = Vec::new();

    for item in outer.iter() {
        let child = from_python_object(&item)?;
        let child_len = child.len() as i64;
        offsets.push(offsets.last().unwrap() + child_len);
        children.push(child);
    }

    // All children must be the same variant — merge them
    let content = merge_contents(children)?;

    Ok(Content::ListOffsetArray(ListOffsetArray {
        offsets: Arc::from(offsets.into_boxed_slice()),
        content: Arc::new(content),
    }))
}

fn merge_contents(mut children: Vec<Content>) -> PyResult<Content> {
    if children.is_empty() {
        // Empty list — default to empty NumpyArray
        return Ok(Content::NumpyArray(NumpyArray {
            data: Arc::from(vec![].into_boxed_slice()),
            shape: vec![0],
            strides: vec![1],
        }));
    }

    // Check if all are NumpyArray — flatten into one
    if children.iter().all(|c| matches!(c, Content::NumpyArray(_))) {
        let flat: Vec<f64> = children.iter().flat_map(|c| match c {
            Content::NumpyArray(a) => a.data.iter().copied().collect::<Vec<_>>(),
            _ => unreachable!(),
        }).collect();
        let len = flat.len();
        return Ok(Content::NumpyArray(NumpyArray {
            data: Arc::from(flat.into_boxed_slice()),
            shape: vec![len],
            strides: vec![1],
        }));
    }

    // All are ListOffsetArray — flatten offsets and contents recursively
    if children.iter().all(|c| matches!(c, Content::ListOffsetArray(_))) {
        let mut merged_offsets: Vec<i64> = vec![0];
        let mut grandchildren: Vec<Content> = Vec::new();

        for child in children.drain(..) {
            if let Content::ListOffsetArray(a) = child {
                let base = *merged_offsets.last().unwrap();
                // skip first offset (always 0) when appending
                for &o in a.offsets.iter().skip(1) {
                    merged_offsets.push(base + o);
                }
                // unwrap the content into grandchildren
                match Arc::try_unwrap(a.content).unwrap_or_else(|arc| (*arc).clone()) {
                    Content::NumpyArray(inner) => {
                        // each element of inner.data is a grandchild
                        for &v in inner.data.iter() {
                            grandchildren.push(Content::NumpyArray(NumpyArray {
                                data: Arc::from(vec![v].into_boxed_slice()),
                                shape: vec![1],
                                strides: vec![1],
                            }));
                        }
                    }
                    other => grandchildren.push(other),
                }
            }
        }

        let merged_content = merge_contents(grandchildren)?;
        return Ok(Content::ListOffsetArray(ListOffsetArray {
            offsets: Arc::from(merged_offsets.into_boxed_slice()),
            content: Arc::new(merged_content),
        }));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "cannot mix list and non-list types in the same array",
    ))
}

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

pub fn content_to_pyobject(c: &Content, py: Python<'_>) -> PyResult<PyObject> {
    match c {
        Content::NumpyArray(a) => {
            let list: Vec<f64> = a.data.to_vec();
            Ok(list.into_pyobject(py)?.into_any().unbind())
        }
        Content::ListOffsetArray(a) => {
            let items: Vec<PyObject> = (0..a.len())
                .map(|i| {
                    let inner = a.slice(i).ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err("slice out of bounds")
                    })?;
                    content_to_pyobject(&inner, py)
                })
                .collect::<PyResult<_>>()?;
            Ok(PyList::new(py, items)?.into_any().unbind())
        }
        Content::RecordArray(r) => {
            let dict = PyDict::new(py);
            for (field, content) in r.fields.iter().zip(r.contents.iter()) {
                let val = content_to_pyobject(content, py)?;
                dict.set_item(field, val)?;
            }
            Ok(dict.into_any().unbind())
        }
        _ => Ok(py.None()),
    }
}