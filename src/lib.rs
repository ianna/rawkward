// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::python::layout::PyIndex;
use crate::python::layout::PyIndexedOptionArray;
use crate::python::layout::PyListOffsetArray;
use crate::python::layout::PyNumpyArray;
use crate::python::layout::PyRecordArray;
use crate::python::layout::PyRegularArray;
#[cfg(feature = "python")]
use pyo3::prelude::*;

mod content;
mod kernels;
mod python;

#[cfg(feature = "python")]
#[pymodule]
fn _rawkward(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyArray>()?;
    m.add_class::<PyListOffsetArray>()?;
    m.add_class::<PyNumpyArray>()?;
    m.add_class::<PyIndex>()?;
    m.add_class::<PyRecordArray>()?;
    m.add_class::<PyRegularArray>()?;
    m.add_class::<PyIndexedOptionArray>()?;

    // m.add_class::<python::PyRecord>()?;
    Ok(())
}
