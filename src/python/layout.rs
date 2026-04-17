// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use pyo3::prelude::*;

#[pyclass(name = "Index", module = "rawkward")]
#[derive(Clone)]
pub struct PyIndex {
    pub data: Vec<i64>,
}

#[pymethods]
impl PyIndex {
    fn __repr__(&self) -> String {
        format!(
            "<Index dtype='int64' len='{}'> {:?} </Index>",
            self.data.len(),
            self.data
        )
    }
}

#[pyclass(name = "NumpyArray", module = "rawkward")]
pub struct PyNumpyArray {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
}

#[pymethods]
impl PyNumpyArray {
    fn __repr__(&self) -> String {
        format!(
            "<NumpyArray dtype='float64' len='{}'> {:?} </NumpyArray>",
            self.data.len(),
            self.data
        )
    }
}

#[pyclass(name = "ListOffsetArray", module = "rawkward")]
pub struct PyListOffsetArray {
    pub offsets: PyIndex,
    pub content: PyObject,
}

#[pymethods]
impl PyListOffsetArray {
    #[getter]
    fn offsets(&self) -> PyIndex {
        self.offsets.clone()
    }

    #[getter]
    fn content(&self, py: Python) -> PyObject {
        self.content.clone_ref(py)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let mut s = String::new();

        s.push_str(&format!(
            "<ListOffsetArray len='{}'>\n",
            self.offsets.data.len() - 1
        ));

        // Offsets
        s.push_str("    <offsets>");
        s.push_str(&format!("{}", self.offsets.__repr__()));
        s.push_str("</offsets>\n");

        // Content
        s.push_str("    <content>\n");

        let content_repr = self.content.bind(py).repr()?.extract::<String>()?;

        for line in content_repr.lines() {
            s.push_str("        ");
            s.push_str(line);
            s.push('\n');
        }

        s.push_str("    </content>\n");
        s.push_str("</ListOffsetArray>");

        Ok(s)
    }
}
