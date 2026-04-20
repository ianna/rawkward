// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct NumpyArray {
    pub data: Arc<[f64]>,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
}

impl NumpyArray {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
