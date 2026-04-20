// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::content::Content;
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
    Some(Content),
    None,
}

impl IndexedOptionArray {
    pub fn new(index: Arc<[i64]>, content: Arc<Content>) -> Self {
        IndexedOptionArray { index, content }
    }

    /// Build from a Vec where None entries map to -1.
    pub fn from_option_vec(values: Vec<Option<Content>>, content: Arc<Content>) -> Self {
        // content must already be assembled; index points into it
        // For simple cases, build a flat index
        let index: Arc<[i64]> = Arc::from(
            values
                .iter()
                .enumerate()
                .map(|(i, v)| if v.is_some() { i as i64 } else { -1 })
                .collect::<Vec<_>>()
                .as_slice(),
        );
        IndexedOptionArray { index, content }
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Get element at position `idx`. Returns None for missing values.
    pub fn get(&self, idx: i64) -> OptionValue {
        let len = self.len() as i64;
        let i = if idx < 0 { len + idx } else { idx };
        if i < 0 || i as usize >= self.len() {
            return OptionValue::None;
        }
        let index_val = self.index[i as usize];
        if index_val < 0 {
            OptionValue::None
        } else {
            use crate::content::regular_array::slice_content;
            let pos = index_val as usize;
            OptionValue::Some(slice_content(&self.content, pos, pos + 1))
        }
    }

    /// Slice this option array by range, returning a new IndexedOptionArray.
    pub fn slice_range(&self, start: usize, stop: usize) -> IndexedOptionArray {
        IndexedOptionArray {
            index: Arc::from(&self.index[start..stop]),
            content: self.content.clone(),
        }
    }

    /// Project a field through the option layer.
    pub fn field(&self, name: &str) -> Option<IndexedOptionArray> {
        match self.content.as_ref() {
            Content::RecordArray(r) => {
                let i = r.fields.iter().position(|f| f == name)?;
                Some(IndexedOptionArray {
                    index: self.index.clone(),
                    content: r.contents[i].clone(),
                })
            }
            _ => None,
        }
    }

    /// Return a boolean mask: true where value is present, false where None.
    pub fn is_valid(&self) -> Vec<bool> {
        self.index.iter().map(|&i| i >= 0).collect()
    }

    /// Return a boolean mask: true where value is None.
    pub fn is_none(&self) -> Vec<bool> {
        self.index.iter().map(|&i| i < 0).collect()
    }

    /// Count non-None values.
    pub fn count_valid(&self) -> usize {
        self.index.iter().filter(|&&i| i >= 0).count()
    }

    /// Flatten: return only the valid (non-None) values as a plain Content.
    /// The returned content has length == count_valid().
    pub fn drop_none(&self) -> Content {
        use crate::content::regular_array::slice_content;

        // Gather valid indices in order
        let valid_indices: Vec<usize> = self
            .index
            .iter()
            .filter(|&&i| i >= 0)
            .map(|&i| i as usize)
            .collect();

        // For NumpyArray content: gather values directly
        match self.content.as_ref() {
            Content::NumpyArray(a) => {
                let data: Vec<f64> = valid_indices.iter().map(|&i| a.data[i]).collect();
                let len = data.len();
                Content::NumpyArray(crate::content::NumpyArray {
                    data: Arc::from(data.into_boxed_slice()),
                    shape: vec![len],
                    strides: vec![1],
                })
            }
            _ => {
                // General case: gather slices
                // This materializes, which is fine for a CPU-only engine
                let slices: Vec<Content> = valid_indices
                    .iter()
                    .map(|&i| slice_content(&self.content, i, i + 1))
                    .collect();
                crate::content::merge_contents_same_type(slices)
                    .unwrap_or_else(|| self.content.as_ref().clone())
            }
        }
    }

    /// Convert to a Python-friendly representation: Vec<Option<Content>>
    pub fn to_option_vec(&self) -> Vec<OptionValue> {
        (0..self.len()).map(|i| self.get(i as i64)).collect()
    }
}
