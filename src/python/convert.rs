// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use std::sync::Arc;

use pyo3::PyObject;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use super::layout::{
    PyIndex, PyIndexedOptionArray, PyListOffsetArray, PyNumpyArray, PyRecordArray, PyRegularArray,
};
use crate::content::indexed_option_array::OptionValue;
use crate::content::{Content, IndexedOptionArray, ListOffsetArray, NumpyArray};

pub fn from_python_object(obj: &Bound<'_, PyAny>) -> PyResult<Content> {
    // Case 1: scalar number → single-element NumpyArray
    if let Ok(val) = obj.extract::<f64>() {
        return Ok(Content::NumpyArray(NumpyArray {
            data: Arc::from(vec![val].into_boxed_slice()),
            shape: vec![1],
            strides: vec![1],
        }));
    }

    // Case 2: flat list of numbers → NumpyArray
    if let Ok(flat) = obj.extract::<Vec<f64>>() {
        let len = flat.len();
        let data = Arc::from(flat.into_boxed_slice());
        return Ok(Content::NumpyArray(NumpyArray {
            data,
            shape: vec![len],
            strides: vec![1],
        }));
    }

    // Case 3: dict → RecordArray (single record, length 1)
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

    // Case 4: list → recurse
    let outer = obj.downcast::<PyList>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "expected a list or dict, got {}",
            obj.get_type()
                .name()
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "unknown".into())
        ))
    })?;
    //    let outer = obj.downcast::<PyList>()...;
    // Check if any item is Python None
    let has_none = outer.iter().any(|item| item.is_none());

    if has_none {
        // Build index and content separately
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
            Content::NumpyArray(NumpyArray {
                data: Arc::from(vec![].into_boxed_slice()),
                shape: vec![0],
                strides: vec![1],
            })
        } else {
            merge_contents(valid_children)?
        };

        return Ok(Content::IndexedOptionArray(IndexedOptionArray {
            index: Arc::from(index.into_boxed_slice()),
            content: Arc::new(content),
        }));
    }

    let mut children: Vec<Content> = Vec::new();
    for item in outer.iter() {
        children.push(from_python_object(&item)?);
    }

    if children
        .iter()
        .all(|c| matches!(c, Content::RecordArray(_)))
    {
        return merge_contents(children);
    }
    // // If children are RecordArrays, merge columns and use row-based offsets
    // if children.iter().all(|c| matches!(c, Content::RecordArray(_))) {
    //     let n = children.len();
    //     let offsets: Vec<i64> = (0..=n as i64).collect();
    //     let content = merge_contents(children)?;
    //     return Ok(Content::ListOffsetArray(ListOffsetArray {
    //         offsets: Arc::from(offsets.into_boxed_slice()),
    //         content: Arc::new(content),
    //     }));
    // }

    // General case: use child.len() for offsets
    let mut offsets: Vec<i64> = vec![0];
    for child in &children {
        offsets.push(offsets.last().unwrap() + child.len() as i64);
    }
    let content = merge_contents(children)?;
    Ok(Content::ListOffsetArray(ListOffsetArray {
        offsets: Arc::from(offsets.into_boxed_slice()),
        content: Arc::new(content),
    }))

    // // Case 4: list → recurse
    // let outer = obj.downcast::<PyList>()
    //     .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
    //         format!("expected a list or dict, got {}",
    //             obj.get_type().name()
    //                 .map(|s| s.to_string())
    //                 .unwrap_or_else(|_| "unknown".into()))
    //     ))?;

    // let mut offsets: Vec<i64> = vec![0];
    // let mut children: Vec<Content> = Vec::new();

    // for item in outer.iter() {
    //     let child = from_python_object(&item)?;
    //     let child_len = child.len() as i64;
    //     offsets.push(offsets.last().unwrap() + child_len);
    //     children.push(child);
    // }

    // let content = merge_contents(children)?;

    // Ok(Content::ListOffsetArray(ListOffsetArray {
    //     offsets: Arc::from(offsets.into_boxed_slice()),
    //     content: Arc::new(content),
    // }))
}

fn merge_contents(mut children: Vec<Content>) -> PyResult<Content> {
    if children.is_empty() {
        return Ok(Content::NumpyArray(NumpyArray {
            data: Arc::from(vec![].into_boxed_slice()),
            shape: vec![0],
            strides: vec![1],
        }));
    }

    // All NumpyArray → flatten into one contiguous buffer
    if children.iter().all(|c| matches!(c, Content::NumpyArray(_))) {
        let flat: Vec<f64> = children
            .iter()
            .flat_map(|c| match c {
                Content::NumpyArray(a) => a.data.iter().copied().collect::<Vec<_>>(),
                _ => unreachable!(),
            })
            .collect();
        let len = flat.len();
        return Ok(Content::NumpyArray(NumpyArray {
            data: Arc::from(flat.into_boxed_slice()),
            shape: vec![len],
            strides: vec![1],
        }));
    }

    // All RecordArray → merge column-wise
    if children
        .iter()
        .all(|c| matches!(c, Content::RecordArray(_)))
    {
        let fields = match &children[0] {
            Content::RecordArray(r) => r.fields.clone(),
            _ => unreachable!(),
        };

        let n = children.len();
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
            .map(|field_items| Ok(Arc::new(merge_contents(field_items)?)))
            .collect::<PyResult<Vec<_>>>()?;

        return Ok(Content::RecordArray(crate::content::RecordArray {
            fields,
            contents,
            length: n,
        }));
    }

    // // All ListOffsetArray → merge offsets and recurse into content
    // if children.iter().all(|c| matches!(c, Content::ListOffsetArray(_))) {
    //     let mut merged_offsets: Vec<i64> = vec![0];
    //     let mut grandchildren: Vec<Content> = Vec::new();

    //     for child in children.drain(..) {
    //         if let Content::ListOffsetArray(a) = child {
    //             let base = *merged_offsets.last().unwrap();
    //             for &o in a.offsets.iter().skip(1) {
    //                 merged_offsets.push(base + o);
    //             }
    //             match Arc::try_unwrap(a.content).unwrap_or_else(|arc| (*arc).clone()) {
    //                 Content::NumpyArray(inner) => {
    //                     for &v in inner.data.iter() {
    //                         grandchildren.push(Content::NumpyArray(NumpyArray {
    //                             data: Arc::from(vec![v].into_boxed_slice()),
    //                             shape: vec![1],
    //                             strides: vec![1],
    //                         }));
    //                     }
    //                 }
    //                 other => grandchildren.push(other),
    //             }
    //         }
    //     }

    //     let merged_content = merge_contents(grandchildren)?;
    //     return Ok(Content::ListOffsetArray(ListOffsetArray {
    //         offsets: Arc::from(merged_offsets.into_boxed_slice()),
    //         content: Arc::new(merged_content),
    //     }));
    // }
    // All children are ListOffsetArray (e.g. outer rows of a 3D array) CHECK !!!!
    // if children.iter().all(|c| matches!(c, Content::ListOffsetArray(_))) {
    //     let mut outer_offsets: Vec<i64> = vec![0];
    //     let mut inner_children: Vec<Content> = Vec::new();

    //     for child in children.drain(..) {
    //         if let Content::ListOffsetArray(a) = child {
    //             let base = *outer_offsets.last().unwrap();
    //             for &o in a.offsets.iter().skip(1) {
    //                 outer_offsets.push(base + o);
    //             }
    //             // Collect the inner lists as separate Content items
    //             let inner_content = Arc::try_unwrap(a.content)
    //                 .unwrap_or_else(|arc| (*arc).clone());
    //             let n = a.offsets.len() - 1;
    //             for i in 0..n {
    //                 let start = a.offsets[i] as usize;
    //                 let stop = a.offsets[i + 1] as usize;
    //                 inner_children.push(
    //                     crate::content::regular_array::slice_content(&inner_content, start, stop)
    //                 );
    //             }
    //         }
    //     }

    //     let merged_inner = merge_contents(inner_children)?;
    //     return Ok(Content::ListOffsetArray(crate::content::ListOffsetArray {
    //         offsets: Arc::from(outer_offsets.into_boxed_slice()),
    //         content: Arc::new(merged_inner),
    //     }));
    // }

    // All ListOffsetArray → merge offsets and recurse into content
    if children
        .iter()
        .all(|c| matches!(c, Content::ListOffsetArray(_)))
    {
        let mut merged_offsets: Vec<i64> = vec![0];
        let mut all_contents: Vec<Content> = Vec::new();

        for child in children.drain(..) {
            if let Content::ListOffsetArray(a) = child {
                let base = *merged_offsets.last().unwrap();
                for &o in a.offsets.iter().skip(1) {
                    merged_offsets.push(base + o);
                }
                // Collect the inner content as-is, don't explode it
                match Arc::try_unwrap(a.content).unwrap_or_else(|arc| (*arc).clone()) {
                    Content::NumpyArray(inner) => {
                        // flat numbers: each value is a separate child
                        for &v in inner.data.iter() {
                            all_contents.push(Content::NumpyArray(NumpyArray {
                                data: Arc::from(vec![v].into_boxed_slice()),
                                shape: vec![1],
                                strides: vec![1],
                            }));
                        }
                    }
                    Content::RecordArray(r) => {
                        // split the record back into per-row RecordArrays
                        for row in 0..r.length {
                            let row_contents: Vec<Arc<Content>> = r
                                .contents
                                .iter()
                                .map(|col| Arc::new(slice_col(col, row, row + 1)))
                                .collect();
                            all_contents.push(Content::RecordArray(crate::content::RecordArray {
                                fields: r.fields.clone(),
                                contents: row_contents,
                                length: 1,
                            }));
                        }
                    }
                    other => all_contents.push(other),
                }
            }
        }

        let merged_content = merge_contents(all_contents)?;
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
    offsets.push(0i64);

    for sub in &outer {
        offsets.push(offsets.last().unwrap() + sub.len() as i64);
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
            let offsets = PyIndex {
                data: a.offsets.to_vec(),
            };
            let content = content_to_python(py, &a.content)?;
            let obj = PyListOffsetArray { offsets, content };
            Py::new(py, obj).map(|p| p.into_bound(py).into_any().unbind())
        }

        // Content::RecordArray(r) => {
        //     let rows: Vec<PyObject> = (0..r.length)
        //         .map(|row| {
        //             let dict = PyDict::new(py);
        //             for (field, col) in r.fields.iter().zip(r.contents.iter()) {
        //                 let val = scalar_to_pyobject(col, row, py)?;
        //                 dict.set_item(field, val)?;
        //             }
        //             Ok(dict.into_any().unbind())
        //         })
        //         .collect::<PyResult<_>>()?;
        //     Ok(PyList::new(py, rows)?.into_any().unbind())
        // }
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
        // Content::RegularArray(a) => {
        //     let items: Vec<PyObject> = (0..a.len())
        //         .map(|i| {
        //             let inner = a.slice_at(i as i64).ok_or_else(|| {
        //                 pyo3::exceptions::PyIndexError::new_err("index out of bounds")
        //             })?;
        //             content_to_python(py, &inner)
        //         })
        //         .collect::<PyResult<_>>()?;
        //     Ok(PyList::new(py, items)?.into_any().unbind())
        // }
        Content::IndexedOptionArray(a) => {
            let index = PyIndex {
                data: a.index.to_vec(),
            };
            let content = content_to_python(py, &a.content)?;
            let obj = PyIndexedOptionArray { index, content };
            Py::new(py, obj).map(|p| p.into_bound(py).into_any().unbind())
        }
        // Content::IndexedOptionArray(a) => {
        //     let items: Vec<PyObject> = (0..a.len())
        //         .map(|i| {
        //             match a.get(i as i64) {
        //                 OptionValue::None => Ok(py.None()),
        //                 OptionValue::Some(c) => content_to_pyobject(&c, py),
        //             }
        //         })
        //         .collect::<PyResult<_>>()?;
        //     Ok(PyList::new(py, items)?.into_any().unbind())
        // }
        _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "layout conversion not implemented",
        )),
    }
}

fn scalar_to_pyobject(col: &Content, row: usize, py: Python<'_>) -> PyResult<PyObject> {
    match col {
        Content::NumpyArray(a) => Ok(a.data[row].into_pyobject(py)?.into_any().unbind()),
        Content::ListOffsetArray(a) => {
            let inner = a
                .slice(row)
                .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("out of bounds"))?;
            content_to_pyobject(&inner, py)
        }
        Content::IndexedOptionArray(a) => {
            use crate::content::indexed_option_array::OptionValue;
            match a.get(row as i64) {
                OptionValue::None => Ok(py.None()),
                OptionValue::Some(c) => content_to_pyobject(&c, py),
            }
        }
        _ => content_to_pyobject(col, py),
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

        Content::IndexedOptionArray(a) => {
            use crate::content::indexed_option_array::OptionValue;
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

fn slice_col(c: &Content, start: usize, stop: usize) -> Content {
    match c {
        Content::NumpyArray(a) => {
            let data: Arc<[f64]> = Arc::from(&a.data[start..stop]);
            let len = data.len();
            Content::NumpyArray(NumpyArray {
                data,
                shape: vec![len],
                strides: vec![1],
            })
        }
        Content::ListOffsetArray(a) => {
            let new_base = a.offsets[start];
            let new_offsets: Arc<[i64]> = Arc::from(
                a.offsets[start..=stop]
                    .iter()
                    .map(|o| o - new_base)
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            Content::ListOffsetArray(ListOffsetArray {
                offsets: new_offsets,
                content: a.content.clone(),
            })
        }
        _ => unimplemented!("slice_col not implemented for this layout"),
    }
}
