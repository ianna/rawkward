// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use pyo3::prelude::*;

// ── Index ─────────────────────────────────────────────────────────────────────

#[pyclass(name = "Index", module = "rawkward")]
#[derive(Clone)]
pub struct PyIndex {
    pub data: Vec<i64>,
}

#[pymethods]
impl PyIndex {
    fn __repr__(&self) -> String {
        let vals: Vec<String> = self.data.iter().map(|v| v.to_string()).collect();
        format!(
            "<Index dtype='int64' len='{}'>[{}]</Index>",
            self.data.len(),
            vals.join(" ")
        )
    }
}

// ── NumpyArray ────────────────────────────────────────────────────────────────

#[pyclass(name = "NumpyArray", module = "rawkward")]
pub struct PyNumpyArray {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
}

#[pymethods]
impl PyNumpyArray {
    fn __repr__(&self) -> String {
        let vals: Vec<String> = self
            .data
            .iter()
            .map(|v| {
                if v.fract() == 0.0 {
                    format!("{v:.1}")
                } else {
                    format!("{v}")
                }
            })
            .collect();
        format!(
            "<NumpyArray dtype='float64' len='{}'>[{}]</NumpyArray>",
            self.data.len(),
            vals.join(" ")
        )
    }
    // fn __repr__(&self) -> String {
    //     let vals: Vec<String> = self.data.iter().map(|v| format!("{v}")).collect();
    //     format!(
    //         "<NumpyArray dtype='float64' len='{}'>[{}]</NumpyArray>",
    //         self.data.len(),
    //         vals.join(" ")
    //     )
    // }
}

// ── RecordArray ───────────────────────────────────────────────────────────────

#[pyclass(name = "RecordArray", module = "rawkward")]
pub struct PyRecordArray {
    pub fields: Vec<String>,
    pub contents: Vec<PyObject>, // one per field
    pub length: usize,
}

#[pymethods]
impl PyRecordArray {
    #[getter]
    fn fields(&self) -> Vec<String> {
        self.fields.clone()
    }

    #[getter]
    fn length(&self) -> usize {
        self.length
    }

    fn content(&self, py: Python, index: usize) -> PyResult<PyObject> {
        self.contents
            .get(index)
            .map(|o| o.clone_ref(py))
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("index out of range"))
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let is_tuple = self.fields.is_empty();
        let mut s = format!(
            "<RecordArray is_tuple='{}' len='{}'>\n",
            is_tuple, self.length
        );

        for (i, (field, content)) in self.fields.iter().zip(self.contents.iter()).enumerate() {
            s.push_str(&format!("    <content index='{i}' field='{field}'>\n"));
            let content_repr = content.bind(py).repr()?.extract::<String>()?;
            for line in content_repr.lines() {
                s.push_str("        ");
                s.push_str(line);
                s.push('\n');
            }
            s.push_str("    </content>\n");
        }

        s.push_str("</RecordArray>");
        Ok(s)
    }
}

// ── ListOffsetArray ───────────────────────────────────────────────────────────

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
        let len = self.offsets.data.len().saturating_sub(1);
        let mut s = format!("<ListOffsetArray len='{len}'>\n");

        s.push_str("    <offsets>");
        s.push_str(&self.offsets.__repr__());
        s.push_str("</offsets>\n");

        s.push_str("    <content>");
        let content_repr = self.content.bind(py).repr()?.extract::<String>()?;
        // indent all lines of nested content
        let indented: Vec<String> = content_repr
            .lines()
            .enumerate()
            .map(|(i, line)| {
                if i == 0 {
                    line.to_string()
                } else {
                    format!("        {line}")
                }
            })
            .collect();
        s.push_str(&indented.join("\n"));
        s.push_str("</content>\n");

        s.push_str("</ListOffsetArray>");
        Ok(s)
    }
}

// ── RegularArray layout ───────────────────────────────────────────────────────

#[pyclass(name = "RegularArray", module = "rawkward")]
pub struct PyRegularArray {
    pub content: PyObject,
    pub size: usize,
    pub length: usize,
}

#[pymethods]
impl PyRegularArray {
    #[getter]
    fn size(&self) -> usize {
        self.size
    }

    #[getter]
    fn length(&self) -> usize {
        self.length
    }

    #[getter]
    fn content(&self, py: Python) -> PyObject {
        self.content.clone_ref(py)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let mut s = format!(
            "<RegularArray size='{}' len='{}'>\n",
            self.size, self.length
        );
        s.push_str("    <content>");
        let content_repr = self.content.bind(py).repr()?.extract::<String>()?;
        let indented: Vec<String> = content_repr
            .lines()
            .enumerate()
            .map(|(i, line)| {
                if i == 0 {
                    line.to_string()
                } else {
                    format!("        {line}")
                }
            })
            .collect();
        s.push_str(&indented.join("\n"));
        s.push_str("</content>\n");
        s.push_str("</RegularArray>");
        Ok(s)
    }
}

// ── IndexedOptionArray layout ─────────────────────────────────────────────────

#[pyclass(name = "IndexedOptionArray", module = "rawkward")]
pub struct PyIndexedOptionArray {
    pub index: PyIndex,
    pub content: PyObject,
}

#[pymethods]
impl PyIndexedOptionArray {
    #[getter]
    fn index(&self) -> PyIndex {
        self.index.clone()
    }

    #[getter]
    fn content(&self, py: Python) -> PyObject {
        self.content.clone_ref(py)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let len = self.index.data.len();
        let mut s = format!("<IndexedOptionArray len='{len}'>\n");

        s.push_str("    <index>");
        s.push_str(&self.index.__repr__());
        s.push_str("</index>\n");

        s.push_str("    <content>");
        let content_repr = self.content.bind(py).repr()?.extract::<String>()?;
        let indented: Vec<String> = content_repr
            .lines()
            .enumerate()
            .map(|(i, line)| {
                if i == 0 {
                    line.to_string()
                } else {
                    format!("        {line}")
                }
            })
            .collect();
        s.push_str(&indented.join("\n"));
        s.push_str("</content>\n");
        s.push_str("</IndexedOptionArray>");
        Ok(s)
    }
}
