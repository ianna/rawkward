// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use std::sync::Arc;

use pyo3::PyObject;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use super::py_layout::{
    PyIndex, PyIndexedOptionArray, PyListOffsetArray, PyNumpyArray, PyRecordArray, PyRegularArray,
};
use crate::content::indexed_option_array::OptionValue;
use crate::content::{Content, IndexedOptionArray, ListOffsetArray, NumpyArray};
use crate::dtype::DType;
use crate::layout::Layout;

// ── Construction from Python ──────────────────────────────────────────────────

pub fn from_python_object(obj: &Bound<'_, PyAny>) -> PyResult<Content> {
    // Case 1: bool must be checked before f64 — Python bool is a subtype of int
    if let Ok(val) = obj.extract::<bool>() {
        return Ok(Content::Bool(NumpyArray {
            data: Arc::from(vec![val].into_boxed_slice()),
            shape: vec![1],
            strides: vec![1],
            dtype: DType::Bool,
        }));
    }

    // Case 2: integer
    if let Ok(val) = obj.extract::<i64>() {
        return Ok(Content::I64(NumpyArray {
            data: Arc::from(vec![val].into_boxed_slice()),
            shape: vec![1],
            strides: vec![1],
            dtype: DType::I64,
        }));
    }

    // Case 3: float
    if let Ok(val) = obj.extract::<f64>() {
        return Ok(Content::F64(NumpyArray {
            data: Arc::from(vec![val].into_boxed_slice()),
            shape: vec![1],
            strides: vec![1],
            dtype: DType::F64,
        }));
    }

    // Case 4: flat list of floats (fast path)
    if let Ok(flat) = obj.extract::<Vec<f64>>() {
        let len = flat.len();
        return Ok(Content::F64(NumpyArray {
            data: Arc::from(flat.into_boxed_slice()),
            shape: vec![len],
            strides: vec![1],
            dtype: DType::F64,
        }));
    }

    // Case 5: dict → single-row RecordArray
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut fields = Vec::new();
        let mut contents = Vec::new();
        for (k, v) in dict.iter() {
            fields.push(k.extract::<String>()?);
            contents.push(Arc::new(from_python_object(&v)?));
        }
        return Ok(Content::RecordArray(crate::content::RecordArray {
            fields,
            contents,
            length: 1,
        }));
    }

    // Case 6: list → recurse
    let outer = obj.downcast::<PyList>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "expected a list or dict, got {}",
            obj.get_type()
                .name()
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "unknown".into())
        ))
    })?;

    // Handle None values → IndexedOptionArray
    let has_none = outer.iter().any(|item| item.is_none());
    if has_none {
        let mut index: Vec<i64> = Vec::with_capacity(outer.len());
        let mut valid_children: Vec<Content> = Vec::new();
        for item in outer.iter() {
            if item.is_none() {
                index.push(-1);
            } else {
                index.push(valid_children.len() as i64);
                valid_children.push(from_python_object(&item)?);
            }
        }
        let content = if valid_children.is_empty() {
            Content::F64(NumpyArray {
                data: Arc::from([] as [f64; 0]),
                shape: vec![0],
                strides: vec![1],
                dtype: DType::F64,
            })
        } else {
            merge_contents(valid_children)?
        };
        return Ok(Content::IndexedOptionArray(IndexedOptionArray {
            index: Arc::from(index.into_boxed_slice()),
            content: Arc::new(content),
        }));
    }

    // Collect children
    let mut children: Vec<Content> = Vec::new();
    for item in outer.iter() {
        children.push(from_python_object(&item)?);
    }

    let mut offsets: Vec<i64> = vec![0];
    for child in &children {
        offsets.push(offsets.last().unwrap() + child.len() as i64);
    }
    let content = merge_contents(children)?;
    Ok(Content::ListOffsetArray(ListOffsetArray {
        offsets: Arc::from(offsets.into_boxed_slice()),
        content: Arc::new(content),
    }))
}

// ── merge_contents ────────────────────────────────────────────────────────────

fn merge_contents(mut children: Vec<Content>) -> PyResult<Content> {
    if children.is_empty() {
        return Ok(Content::F64(NumpyArray {
            data: Arc::from([] as [f64; 0]),
            shape: vec![0],
            strides: vec![1],
            dtype: DType::F64,
        }));
    }

    if children.iter().all(|c| matches!(c, Content::Bool(_))) {
        let flat: Vec<bool> = children
            .iter()
            .flat_map(|c| match c {
                Content::Bool(a) => a.data.iter().copied().collect::<Vec<_>>(),
                _ => unreachable!(),
            })
            .collect();
        let len = flat.len();
        return Ok(Content::Bool(NumpyArray {
            data: Arc::from(flat.into_boxed_slice()),
            shape: vec![len],
            strides: vec![1],
            dtype: DType::Bool,
        }));
    }

    if children.iter().all(|c| matches!(c, Content::I64(_))) {
        let flat: Vec<i64> = children
            .iter()
            .flat_map(|c| match c {
                Content::I64(a) => a.data.iter().copied().collect::<Vec<_>>(),
                _ => unreachable!(),
            })
            .collect();
        let len = flat.len();
        return Ok(Content::I64(NumpyArray {
            data: Arc::from(flat.into_boxed_slice()),
            shape: vec![len],
            strides: vec![1],
            dtype: DType::I64,
        }));
    }

    if children.iter().all(|c| matches!(c, Content::F64(_))) {
        let flat: Vec<f64> = children
            .iter()
            .flat_map(|c| match c {
                Content::F64(a) => a.data.iter().copied().collect::<Vec<_>>(),
                _ => unreachable!(),
            })
            .collect();
        let len = flat.len();
        return Ok(Content::F64(NumpyArray {
            data: Arc::from(flat.into_boxed_slice()),
            shape: vec![len],
            strides: vec![1],
            dtype: DType::F64,
        }));
    }

    // All RecordArray — merge column-wise
    if children
        .iter()
        .all(|c| matches!(c, Content::RecordArray(_)))
    {
        let fields = match &children[0] {
            Content::RecordArray(r) => r.fields.clone(),
            _ => unreachable!(),
        };
        let total_rows: usize = children
            .iter()
            .map(|c| match c {
                Content::RecordArray(r) => r.length,
                _ => unreachable!(),
            })
            .sum();
        let mut per_field: Vec<Vec<Content>> = vec![Vec::new(); fields.len()];
        for child in children.drain(..) {
            if let Content::RecordArray(r) = child {
                for (i, c) in r.contents.into_iter().enumerate() {
                    per_field[i].push(Arc::try_unwrap(c).unwrap_or_else(|arc| (*arc).clone()));
                }
            }
        }
        let contents = per_field
            .into_iter()
            .map(|items| Ok(Arc::new(merge_contents(items)?)))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Content::RecordArray(crate::content::RecordArray {
            fields,
            contents,
            length: total_rows,
        }));
    }

    // All ListOffsetArray — re-expand and re-merge sub-lists
    if children
        .iter()
        .all(|c| matches!(c, Content::ListOffsetArray(_)))
    {
        let mut merged_offsets: Vec<i64> = vec![0];
        let mut inner_items: Vec<Content> = Vec::new();

        for child in children.drain(..) {
            if let Content::ListOffsetArray(a) = child {
                let n = a.offsets.len().saturating_sub(1);
                let inner = Arc::try_unwrap(a.content).unwrap_or_else(|arc| (*arc).clone());
                for i in 0..n {
                    let start = a.offsets[i] as usize;
                    let stop = a.offsets[i + 1] as usize;
                    inner_items.push((*inner.slice_arc(start, stop)).clone());
                }
            }
        }
        for item in &inner_items {
            merged_offsets.push(merged_offsets.last().unwrap() + item.len() as i64);
        }
        let merged_inner = merge_contents(inner_items)?;
        return Ok(Content::ListOffsetArray(ListOffsetArray {
            offsets: Arc::from(merged_offsets.into_boxed_slice()),
            content: Arc::new(merged_inner),
        }));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "cannot mix list and non-list types in the same array",
    ))
}

// ── Legacy helper ─────────────────────────────────────────────────────────────

pub fn from_python_nested_list(obj: &Bound<'_, PyAny>) -> PyResult<Content> {
    let outer: Vec<Vec<f64>> = obj.extract()?;
    let mut offsets = vec![0i64];
    for sub in &outer {
        offsets.push(offsets.last().unwrap() + sub.len() as i64);
    }
    let flat: Vec<f64> = outer.into_iter().flatten().collect();
    let flat_len = flat.len();
    Ok(Content::ListOffsetArray(ListOffsetArray {
        offsets: Arc::from(offsets.into_boxed_slice()),
        content: Arc::new(Content::F64(NumpyArray {
            data: Arc::from(flat.into_boxed_slice()),
            shape: vec![flat_len],
            strides: vec![1],
            dtype: DType::F64,
        })),
    }))
}

// ── Layout objects ────────────────────────────────────────────────────────────

pub fn content_to_python(py: Python, c: &Content) -> PyResult<PyObject> {
    match c {
        Content::Bool(a) => {
            let obj = PyNumpyArray {
                data: a.data.iter().map(|&x| x as u8 as f64).collect(),
                shape: a.shape.clone(),
                strides: a.strides.clone(),
            };
            Py::new(py, obj).map(|p| p.into_bound(py).into_any().unbind())
        }
        Content::I64(a) => {
            let obj = PyNumpyArray {
                data: a.data.iter().map(|&x| x as f64).collect(),
                shape: a.shape.clone(),
                strides: a.strides.clone(),
            };
            Py::new(py, obj).map(|p| p.into_bound(py).into_any().unbind())
        }
        Content::F64(a) => {
            let obj = PyNumpyArray {
                data: a.data.to_vec(),
                shape: a.shape.clone(),
                strides: a.strides.clone(),
            };
            Py::new(py, obj).map(|p| p.into_bound(py).into_any().unbind())
        }
        Content::ListOffsetArray(a) => {
            let offsets = PyIndex {
                data: a.offsets.to_vec(),
            };
            let content = content_to_python(py, &a.content)?;
            let obj = PyListOffsetArray { offsets, content };
            Py::new(py, obj).map(|p| p.into_bound(py).into_any().unbind())
        }
        Content::RecordArray(r) => {
            let contents: Vec<PyObject> = r
                .contents
                .iter()
                .map(|c| content_to_python(py, c))
                .collect::<PyResult<_>>()?;
            let obj = PyRecordArray {
                fields: r.fields.clone(),
                contents,
                length: r.length,
            };
            Py::new(py, obj).map(|p| p.into_bound(py).into_any().unbind())
        }
        Content::RegularArray(a) => {
            let content = content_to_python(py, &a.content)?;
            let obj = PyRegularArray {
                content,
                size: a.size,
                length: a.length,
            };
            Py::new(py, obj).map(|p| p.into_bound(py).into_any().unbind())
        }
        Content::IndexedOptionArray(a) => {
            let index = PyIndex {
                data: a.index.to_vec(),
            };
            let content = content_to_python(py, &a.content)?;
            let obj = PyIndexedOptionArray { index, content };
            Py::new(py, obj).map(|p| p.into_bound(py).into_any().unbind())
        }
        _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "layout conversion not implemented",
        )),
    }
}

// ── tolist ────────────────────────────────────────────────────────────────────

pub fn content_to_pyobject(c: &Content, py: Python<'_>) -> PyResult<PyObject> {
    match c {
        Content::Bool(a) => {
            let list: Vec<bool> = a.data.to_vec();
            Ok(list.into_pyobject(py)?.into_any().unbind())
        }
        Content::I64(a) => {
            let list: Vec<i64> = a.data.to_vec();
            Ok(list.into_pyobject(py)?.into_any().unbind())
        }
        Content::F64(a) => {
            let list: Vec<f64> = a.data.to_vec();
            Ok(list.into_pyobject(py)?.into_any().unbind())
        }
        Content::ListOffsetArray(a) => {
            let items: Vec<PyObject> = (0..a.len())
                .map(|i| {
                    let inner = (*a
                        .content
                        .slice_arc(a.offsets[i] as usize, a.offsets[i + 1] as usize))
                    .clone();
                    content_to_pyobject(&inner, py)
                })
                .collect::<PyResult<_>>()?;
            Ok(PyList::new(py, items)?.into_any().unbind())
        }
        Content::RecordArray(r) => {
            let rows: Vec<PyObject> = (0..r.length)
                .map(|row| {
                    let dict = PyDict::new(py);
                    for (field, col) in r.fields.iter().zip(r.contents.iter()) {
                        let val = scalar_to_pyobject(col, row, py)?;
                        dict.set_item(field, val)?;
                    }
                    Ok(dict.into_any().unbind())
                })
                .collect::<PyResult<_>>()?;
            Ok(PyList::new(py, rows)?.into_any().unbind())
        }
        Content::RegularArray(a) => {
            // Bug 1 fix: use get(i) instead of nonexistent slice_at()
            let items: Vec<PyObject> = (0..a.length)
                .map(|i| {
                    let inner = a.get(i).ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err("index out of bounds")
                    })?;
                    content_to_pyobject(&inner, py)
                })
                .collect::<PyResult<_>>()?;
            Ok(PyList::new(py, items)?.into_any().unbind())
        }
        Content::IndexedOptionArray(a) => {
            let items: Vec<PyObject> = (0..a.len())
                .map(|i| match a.get(i as i64) {
                    OptionValue::None => Ok(py.None()),
                    OptionValue::Some(c) => content_to_pyobject(&c, py),
                })
                .collect::<PyResult<_>>()?;
            Ok(PyList::new(py, items)?.into_any().unbind())
        }
        _ => Ok(py.None()),
    }
}

fn scalar_to_pyobject(col: &Content, row: usize, py: Python<'_>) -> PyResult<PyObject> {
    match col {
        Content::Bool(a) => Ok(a.data[row]
            .into_pyobject(py)?
            .clone()
            .as_any()
            .clone()
            .unbind()),
        Content::I64(a) => Ok(a.data[row].into_pyobject(py)?.into_any().unbind()),
        Content::F64(a) => Ok(a.data[row].into_pyobject(py)?.into_any().unbind()),
        Content::ListOffsetArray(a) => {
            // Bug 2 fix: use get(row) → Option<Content>, not slice().ok_or_else()
            let inner = a
                .get(row)
                .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("out of bounds"))?;
            content_to_pyobject(&inner, py)
        }
        Content::IndexedOptionArray(a) => {
            // Bug 3 fix: use a.get(row as i64) → OptionValue, not a.index.get(row as i64)
            match a.get(row as i64) {
                OptionValue::None => Ok(py.None()),
                OptionValue::Some(c) => content_to_pyobject(&c, py),
            }
        }
        _ => content_to_pyobject(col, py),
    }
}
