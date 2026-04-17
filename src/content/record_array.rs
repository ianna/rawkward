// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::sync::Arc;

use super::Content;

#[derive(Clone, Debug)]
pub struct RecordArray {
    pub fields: Vec<String>,
    pub contents: Vec<Arc<Content>>,
    pub length: usize,
}

impl RecordArray {
    pub fn len(&self) -> usize {
        self.contents.len()
    }

    pub fn get_field(&self, i: usize) -> Option<&Arc<Content>> {
        self.contents.get(i)
    }
}