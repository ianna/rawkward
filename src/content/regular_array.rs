// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::content::Content;
use crate::layout::Layout;
use std::sync::Arc;

/// RegularArray describes lists that all have the same fixed length (`size`).
/// The underlying `content` is a flat buffer; list i occupies content[i*size .. (i+1)*size].
///
/// If size == 0, length is given explicitly via `zeros_length`.
///
/// Equivalent to Arrow FixedSizeList.
#[derive(Clone, Debug)]
pub struct RegularArray {
    pub content: Arc<Content>,
    pub size: usize,
    pub length: usize, // zeros_length when size == 0
}

impl RegularArray {
    pub fn new(content: Arc<Content>, size: usize, zeros_length: usize) -> Self {
        let length = content.len().checked_div(size).unwrap_or(zeros_length);
        RegularArray {
            content,
            size,
            length,
        }
    }

    pub fn get(&self, index: usize) -> Option<Content> {
        if index >= self.length {
            return None;
        }
        let start = index * self.size;
        let stop = start + self.size;
        Some((*self.content.slice_arc(start, stop)).clone())
    }
}

impl Layout for RegularArray {
    fn len(&self) -> usize {
        self.length
    }

    fn getitem(&self, index: usize) -> Arc<dyn Layout> {
        let start = index * self.size;
        let stop = (index + 1) * self.size;
        self.content.slice_arc(start, stop)
    }

    fn slice(&self, start: usize, stop: usize) -> Arc<dyn Layout> {
        let content_start = start * self.size;
        let content_stop = stop * self.size;
        Arc::new(RegularArray {
            content: self.content.slice_arc(content_start, content_stop),
            size: self.size,
            length: stop - start,
        })
    }

    fn typetracer(&self) -> Arc<dyn Layout> {
        Arc::new(RegularArray {
            content: self.content.typetracer_arc(),
            size: self.size,
            length: 0,
        })
    }
}
