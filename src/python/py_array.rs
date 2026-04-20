// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

#![cfg(feature = "python")]

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use crate::content::{Content, NumpyArray};
use crate::kernels::{slice, slice_range, Slice, SliceError};
use crate::python::convert::{content_to_python, content_to_pyobject, from_python_nested_list, from_python_object};
use crate::content::indexed_option_array::OptionValue;

// ── helpers (module-level, not exposed to Python) ────────────────────────────

fn content_ndim(c: &Content) -> usize {
    match c {
        Content::NumpyArray(_) => 1,
        Content::ListOffsetArray(a) => 1 + content_ndim(&a.content),
        Content::RegularArray(a) => 1 + content_ndim(&a.content),
        Content::RecordArray(_) => 1,  // records don't add a dimension
        Content::IndexedOptionArray(a) => content_ndim(&a.content),
        _ => 1,
    }
}
// fn content_ndim(c: &Content) -> usize {
//     match c {
//         Content::NumpyArray(_) => 1,
//         Content::ListOffsetArray(a) => 1 + content_ndim(&a.content),
//         Content::RecordArray(_) => 1,
//         _ => 1,
//     }
// }

fn content_nbytes(c: &Content) -> usize {
    match c {
        Content::NumpyArray(a) => a.data.len() * std::mem::size_of::<f64>(),
        Content::ListOffsetArray(a) => {
            a.offsets.len() * std::mem::size_of::<i64>() + content_nbytes(&a.content)
        }
        Content::RecordArray(r) => r.contents.iter().map(|c| content_nbytes(c)).sum(),
        _ => 0,
    }
}

// fn fmt_content(c: &Content) -> String {
//     match c {
//         Content::NumpyArray(a) => {
//             let items: Vec<String> = a.data.iter().map(|x| format!("{x}")).collect();
//             format!("[{}]", items.join(", "))
//         }
//         Content::ListOffsetArray(a) => {
//             let items: Vec<String> = (0..a.len())
//                 .map(|i| a.slice(i).map(|c| fmt_content(&c)).unwrap_or("?".into()))
//                 .collect();
//             format!("[{}]", items.join(", "))
//         }
//         Content::RecordArray(r) => {
//             if r.fields.is_empty() {
//                 format!("(len={})", r.length)
//             } else {
//                 let pairs: Vec<String> = r.fields.iter().zip(r.contents.iter())
//                     .map(|(f, c)| format!("{f}: {}", fmt_content(c)))
//                     .collect();
//                 format!("{{{}}}", pairs.join(", "))
//             }
//         }
//         _ => "...".into(),
//     }
// }
fn fmt_content(c: &Content) -> String {
    match c {
        Content::NumpyArray(a) => {
            let items: Vec<String> = a.data.iter().map(|x| format!("{x}")).collect();
            format!("[{}]", items.join(", "))
        }
        Content::ListOffsetArray(a) => {
            let items: Vec<String> = (0..a.len())
                .map(|i| a.slice(i).map(|c| fmt_content(&c)).unwrap_or("?".into()))
                .collect();
            format!("[{}]", items.join(", "))
        }
        Content::RecordArray(r) => {
            // Iterate row by row
            let rows: Vec<String> = (0..r.length)
                .map(|row| {
                    let pairs: Vec<String> = r.fields.iter().zip(r.contents.iter())
                        .map(|(f, col)| {
                            let val = fmt_scalar(col, row);
                            format!("{f}: {val}")
                        })
                        .collect();
                    format!("{{{}}}", pairs.join(", "))
                })
                .collect();
            format!("[{}]", rows.join(", "))
        }
        
        Content::RegularArray(a) => {
            let items: Vec<String> = (0..a.len())
                .map(|i| a.slice_at(i as i64).map(|c| fmt_preview(&c, 20)).unwrap_or("?".into()))
                .collect();
            format!("[{}]", items.join(", "))
        }

        Content::IndexedOptionArray(a) => {
            let items: Vec<String> = (0..a.len())
                .map(|i| match a.get(i as i64) {
                    OptionValue::None => "None".into(),
                    OptionValue::Some(c) => fmt_preview(&c, 20),
                })
                .collect();
            format!("[{}]", items.join(", "))
        }

        _ => "...".into(),
    }
}

// fn fmt_scalar(col: &Content, row: usize) -> String {
//     match col {
//         Content::NumpyArray(a) => {
//             if row < a.data.len() {
//                 format!("{}", a.data[row])
//             } else {
//                 "?".into()
//             }
//         }
//         Content::ListOffsetArray(a) => {
//             a.slice(row).map(|c| fmt_content(&c)).unwrap_or("?".into())
//         }
//         _ => "...".into(),
//     }
// }

fn content_type_str(c: &Content) -> String {
    match c {
        Content::NumpyArray(_) => "float64".into(),
        Content::ListOffsetArray(a) => format!("var * {}", content_type_str(&a.content)),
        Content::RecordArray(r) => {
            let fields: Vec<String> = r.fields.iter().zip(r.contents.iter())
                .map(|(f, c)| format!("{f}: {}", content_type_str(c)))
                .collect();
            format!("{{{}}}", fields.join(", "))
        }
        Content::RegularArray(a) => format!("{} * {}", a.size, content_type_str(&a.content)),
        Content::IndexedOptionArray(a) => format!("option[{}]", content_type_str(&a.content)),

        _ => "unknown".into(),
    }
}

fn fmt_preview(c: &Content, limit: usize) -> String {
    match c {
        Content::NumpyArray(a) => {
            let items: Vec<String> = a.data.iter().take(limit).map(|x| format!("{x}")).collect();
            let ellipsis = if a.data.len() > limit { ", ..." } else { "" };
            format!("[{}{}]", items.join(", "), ellipsis)
        }
        Content::ListOffsetArray(a) => {
            let n = a.len();
            let show = n.min(limit);
            let mut parts: Vec<String> = (0..show)
                .map(|i| a.slice(i).map(|c| fmt_preview(&c, limit)).unwrap_or("?".into()))
                .collect();
            if n > show { parts.push("...".into()); }
            format!("[{}]", parts.join(", "))
        }
        Content::RecordArray(r) => {
            // single record display — used when we've already sliced to one row
            let pairs: Vec<String> = r.fields.iter().zip(r.contents.iter())
                .map(|(f, col)| format!("{f}: {}", fmt_scalar(col, 0)))
                .collect();
            format!("{{{}}}", pairs.join(", "))
        }
        _ => "...".into(),
    }
}

// fn fmt_scalar(col: &Content, row: usize) -> String {
//     match col {
//         Content::NumpyArray(a) => {
//             if row < a.data.len() { format!("{}", a.data[row]) } else { "?".into() }
//         }
//         Content::ListOffsetArray(a) => {
//             a.slice(row).map(|c| fmt_preview(&c, 20)).unwrap_or("?".into())
//         }
//         _ => "...".into(),
//     }
// }

fn fmt_scalar(col: &Content, row: usize) -> String {
    match col {
        Content::NumpyArray(a) => {
            if row < a.data.len() { format!("{}", a.data[row]) } else { "?".into() }
        }
        Content::ListOffsetArray(a) => {
            // slice gives the sub-list at this row
            a.slice(row).map(|c| fmt_preview(&c, 20)).unwrap_or("?".into())
        }
        Content::RecordArray(r) => {
            // single row of a nested record
            let pairs: Vec<String> = r.fields.iter().zip(r.contents.iter())
                .map(|(f, col)| format!("{f}: {}", fmt_scalar(col, row)))
                .collect();
            format!("{{{}}}", pairs.join(", "))
        }
        _ => "...".into(),
    }
}

// ── PyArray ───────────────────────────────────────────────────────────────────

#[pyclass(name = "Array", module = "rawkward")]
pub struct PyArray {
    pub(crate) inner: Arc<Content>,
}

#[pymethods]
impl PyArray {
    // ── constructor ──────────────────────────────────────────────────────────
    #[new]
    fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let content = crate::python::convert::from_python_object(obj)?;
        Ok(PyArray { inner: Arc::new(content) })
    }

    // ── layout (low-level view) ──────────────────────────────────────────────

    #[getter]
    pub fn layout(&self, py: Python) -> PyResult<PyObject> {
        content_to_python(py, &self.inner)
    }

    // ── shape / structure properties ─────────────────────────────────────────

    #[getter]
    fn ndim(&self) -> usize {
        content_ndim(&self.inner)
    }

    #[getter]
    fn nbytes(&self) -> usize {
        content_nbytes(&self.inner)
    }

    #[getter]
    fn fields(&self) -> Vec<String> {
        match self.inner.as_ref() {
            Content::RecordArray(r) => r.fields.clone(),
            _ => vec![],
        }
    }

    #[getter]
    fn is_tuple(&self) -> bool {
        match self.inner.as_ref() {
            Content::RecordArray(r) => r.fields.is_empty(),
            _ => false,
        }
    }

    // ── conversion ───────────────────────────────────────────────────────────

    fn tolist(&self, py: Python<'_>) -> PyResult<PyObject> {
        content_to_pyobject(&self.inner, py)
    }

    fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.tolist(py)
    }

    // ── sequence protocol ────────────────────────────────────────────────────

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<PyObject> {
        let len = slf.inner.len();
        let items: Vec<PyObject> = (0..len)
            .map(|i| {
                let s = Slice::Index(i as i64);
                let out = slice(&slf.inner, &s)
                    .map_err(|e: SliceError| pyo3::exceptions::PyIndexError::new_err(e.to_string()))?;
                let arr = PyArray { inner: Arc::new(out) };
                Py::new(py, arr).map(|p| p.into_bound(py).into_any().unbind())
            })
            .collect::<PyResult<_>>()?;
        let list = PyList::new(py, items)?;
        // Return iter(list) — a Python list_iterator
        list.call_method0("__iter__")
            .map(|it| it.into_pyobject(py).unwrap().into_any().unbind())
    }
    // fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<PyObject> {
    //     let len = slf.inner.len();
    //     let items: Vec<PyObject> = (0..len)
    //         .map(|i| {
    //             let s = Slice::Index(i as i64);
    //             let out = slice(&slf.inner, &s)
    //                 .map_err(|e: SliceError| pyo3::exceptions::PyIndexError::new_err(e.to_string()))?;
    //             let arr = PyArray { inner: Arc::new(out) };
    //             Py::new(py, arr).map(|p| p.into_bound(py).into_any().unbind())
    //         })
    //         .collect::<PyResult<_>>()?;
    //     Ok(PyList::new(py, items)?.into_pyobject(py)?.into_any().unbind())
    // }

    fn __getitem__(&self, idx: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        // integer index
        if let Ok(i) = idx.extract::<isize>() {
            let s = Slice::Index(i as i64);
            let out = slice(&self.inner, &s)
                .map_err(|e: SliceError| pyo3::exceptions::PyIndexError::new_err(e.to_string()))?;
            return Ok(PyArray { inner: Arc::new(out) });
        }

        // slice object
        if let Ok(py_slice) = idx.downcast::<pyo3::types::PySlice>() {
            let indices = py_slice.indices((self.inner.len() as i64).try_into().unwrap())?;
            let start = indices.start as usize;
            let stop = indices.stop as usize;

            if indices.step != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err("slice step must be 1"));
            }

            let out = slice_range(&self.inner, start, stop)
                .map_err(|e: SliceError| pyo3::exceptions::PyIndexError::new_err(e.to_string()))?;
            return Ok(PyArray { inner: Arc::new(out) });
        }

        // string → field access
        if let Ok(field) = idx.extract::<String>() {
            return self.get_field_by_name(&field);
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "index must be int, slice, or str",
        ))
    }

    // ── attribute access (dot notation for record fields) ────────────────────

    fn __getattr__(&self, name: &str) -> PyResult<PyArray> {
        self.get_field_by_name(name)
    }

    fn __dir__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut names: Vec<String> = vec![
            "__len__".into(), "__iter__".into(), "__getitem__".into(),
            "__repr__".into(), "__str__".into(),
            "tolist".into(), "to_list".into(), "to_numpy".into(),
            "show".into(),
            "layout".into(), "ndim".into(), "nbytes".into(),
            "fields".into(), "is_tuple".into(),
        ];
        if let Content::RecordArray(r) = self.inner.as_ref() {
            names.extend(r.fields.clone());
        }
        Ok(PyList::new(py, names)?.into_pyobject(py)?.into_any().unbind())
    }

    // ── display ──────────────────────────────────────────────────────────────

    // fn __repr__(&self) -> PyResult<String> {
    //     Ok(format!("Array({})", fmt_content(&self.inner)))
    // }
    fn __repr__(&self) -> PyResult<String> {
        let type_str = content_type_str(&self.inner);
        let preview = fmt_preview(&self.inner, 2);
        Ok(format!("<Array {preview} type='{type_str}'>"))
    }


    // fn __str__(&self) -> PyResult<String> {
    //     Ok(fmt_content(&self.inner))
    // }

    // fn show(&self) -> PyResult<()> {
    //     println!("{}", fmt_content(&self.inner));
    //     Ok(())
    // }
    fn __str__(&self) -> PyResult<String> {
        Ok(fmt_preview(&self.inner, 20))
    }

    fn show(&self) -> PyResult<()> {
        println!("{}", fmt_preview(&self.inner, 20));
        Ok(())
    }

    // ── numpy interop ─────────────────────────────────────────────────────────

    fn to_numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self.inner.as_ref() {
            Content::NumpyArray(a) => {
                let numpy = py.import("numpy")?;
                let list: Vec<f64> = a.data.to_vec();
                numpy.call_method1("array", (list,))
                    .map(|a| a.into_pyobject(py).unwrap().into_any().unbind())
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "cannot convert non-flat array to numpy without allowing missing",
            )),
        }
    }
}

// ── private helpers ───────────────────────────────────────────────────────────

impl PyArray {
    fn get_field_by_name(&self, name: &str) -> PyResult<PyArray> {
        match self.inner.as_ref() {
            Content::RecordArray(r) => {
                if let Some(i) = r.fields.iter().position(|f| f == name) {
                    Ok(PyArray { inner: r.contents[i].clone() })
                } else {
                    Err(pyo3::exceptions::PyAttributeError::new_err(
                        format!("no field '{name}'"),
                    ))
                }
            }
            _ => Err(pyo3::exceptions::PyAttributeError::new_err(
                format!("no attribute '{name}'"),
            )),
        }
    }
}