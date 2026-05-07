// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::layout::Layout;
use std::sync::Arc;

use super::Content;

#[derive(Clone, Debug)]
pub struct RecordArray {
    pub fields: Vec<String>,
    pub contents: Vec<Arc<Content>>,
    pub length: usize,
}

impl RecordArray {
    pub fn new(fields: Vec<String>, contents: Vec<Arc<Content>>) -> Self {
        let length = contents.first().map_or(0, |c| c.len());
        for col in &contents {
            assert_eq!(
                col.len(),
                length,
                "all RecordArray records must have the same length"
            );
        }
        RecordArray {
            fields,
            contents,
            length,
        }
    }

    pub fn get_field(&self, i: usize) -> Option<&Arc<Content>> {
        self.contents.get(i)
    }
}

impl Layout for RecordArray {
    fn len(&self) -> usize {
        self.length
    }

    fn getitem(&self, index: usize) -> Arc<dyn Layout> {
        let contents: Vec<Arc<Content>> = self
            .contents
            .iter()
            .map(|col| col.slice_arc(index, index + 1))
            .collect();
        Arc::new(RecordArray {
            fields: self.fields.clone(),
            contents,
            length: 1,
        })
    }

    fn slice(&self, start: usize, stop: usize) -> Arc<dyn Layout> {
        let contents: Vec<Arc<Content>> = self
            .contents
            .iter()
            .map(|col| col.slice_arc(start, stop))
            .collect();
        Arc::new(RecordArray {
            fields: self.fields.clone(),
            contents,
            length: stop - start,
        })
    }

    fn typetracer(&self) -> Arc<dyn Layout> {
        let contents: Vec<Arc<Content>> = self
            .contents
            .iter()
            .map(|col| col.typetracer_arc())
            .collect();
        Arc::new(RecordArray {
            fields: self.fields.clone(),
            contents,
            length: 0,
        })
    }
}
