// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::sync::Arc;
use crate::content::Content;
use crate::content::IndexedOptionArray;

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
        let length = if size == 0 {
            zeros_length
        } else {
            content.len() / size
        };
        RegularArray { content, size, length }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Return the sub-content for list at index `idx` (negative indexing supported).
    pub fn slice_at(&self, idx: i64) -> Option<Content> {
        let len = self.length as i64;
        let i = if idx < 0 { len + idx } else { idx };
        if i < 0 || i >= len {
            return None;
        }
        let i = i as usize;
        let start = i * self.size;
        let stop = (i + 1) * self.size;
        Some(slice_content(&self.content, start, stop))
    }

    /// Return a sub-range of this RegularArray (row slice).
    pub fn slice_range(&self, start: usize, stop: usize) -> RegularArray {
        let content_start = start * self.size;
        let content_stop = stop * self.size;
        let new_content = Arc::new(slice_content(&self.content, content_start, content_stop));
        RegularArray {
            content: new_content,
            size: self.size,
            length: stop - start,
        }
    }

    /// Project a field through this RegularArray (for records inside regular lists).
    pub fn field(&self, name: &str) -> Option<RegularArray> {
        match self.content.as_ref() {
            Content::RecordArray(r) => {
                let i = r.fields.iter().position(|f| f == name)?;
                Some(RegularArray {
                    content: r.contents[i].clone(),
                    size: self.size,
                    length: self.length,
                })
            }
            _ => None,
        }
    }
}

/// Slice a Content from start..stop (used by both RegularArray and slicing kernels).
pub(crate) fn slice_content(c: &Content, start: usize, stop: usize) -> Content {
    use crate::content::{NumpyArray, ListOffsetArray, RecordArray};
    use std::sync::Arc;

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
            let inner_start = new_base as usize;
            let inner_stop = a.offsets[stop] as usize;
            let new_offsets: Arc<[i64]> = Arc::from(
                a.offsets[start..=stop]
                    .iter()
                    .map(|o| o - new_base)
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            Content::ListOffsetArray(ListOffsetArray {
                offsets: new_offsets,
                content: Arc::new(slice_content(&a.content, inner_start, inner_stop)),
            })
        }
        Content::RegularArray(a) => {
            Content::RegularArray(a.slice_range(start, stop))
        }
        Content::RecordArray(r) => {
            let contents: Vec<Arc<Content>> = r.contents.iter()
                .map(|col| Arc::new(slice_content(col, start, stop)))
                .collect();
            Content::RecordArray(RecordArray {
                fields: r.fields.clone(),
                contents,
                length: stop - start,
            })
        }

        Content::IndexedOptionArray(a) => {
            Content::IndexedOptionArray(IndexedOptionArray {
                index: Arc::from(&a.index[start..stop]),
                content: a.content.clone(),
            })
        }
        _ => unimplemented!("slice_content not implemented for this layout"),
    }
}
