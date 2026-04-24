// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::content::Content;
use crate::layout::Layout;
use std::sync::Arc;

/// IndexedOptionArray represents optional (nullable) data.
///
/// `index` is a Vec<i64> of the same length as the array. Each value either:
///   - is >= 0: points into `content` at that position (present value)
///   - is < 0 (conventionally -1): represents None / missing
///
/// This is the most general option type in awkward-array.
/// Equivalent to: if index[i] < 0 { None } else { content[index[i]] }

#[derive(Clone, Debug)]
pub struct IndexedOptionArray {
    pub index: Arc<[i64]>,
    pub content: Arc<Content>,
}

/// The result of indexing into an IndexedOptionArray at a single position.
#[derive(Debug)]
pub enum OptionValue {
    None,
    Some(Content),
}

impl IndexedOptionArray {
    pub fn new(index: Arc<[i64]>, content: Arc<Content>) -> Self {
        IndexedOptionArray { index, content }
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn get(&self, i: i64) -> OptionValue {
        let idx = if i < 0 {
            self.index.len() as i64 + i
        } else {
            i
        };
        if idx < 0 || idx as usize >= self.index.len() {
            return OptionValue::None;
        }
        let index_val = self.index[idx as usize];
        if index_val < 0 {
            OptionValue::None
        } else {
            let pos = index_val as usize;
            OptionValue::Some((*self.content.slice_arc(pos, pos + 1)).clone())
        }
    }
}

impl Layout for IndexedOptionArray {
    fn len(&self) -> usize {
        self.index.len()
    }

    fn getitem(&self, index: usize) -> Arc<dyn Layout> {
        match self.get(index as i64) {
            OptionValue::None => Arc::new(IndexedOptionArray {
                index: Arc::from([-1i64] as [i64; 1]),
                content: self.content.clone(),
            }),
            OptionValue::Some(c) => Arc::new(c),
        }
    }

    fn slice(&self, start: usize, stop: usize) -> Arc<dyn Layout> {
        Arc::new(IndexedOptionArray {
            index: Arc::from(self.index[start..stop].to_vec().into_boxed_slice()),
            content: self.content.clone(),
        })
    }

    fn typetracer(&self) -> Arc<dyn Layout> {
        Arc::new(IndexedOptionArray {
            index: Arc::from([] as [i64; 0]),
            content: self.content.typetracer_arc(),
        })
    }
}
