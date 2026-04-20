// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

pub mod numpy_array;
pub mod list_offset_array;
pub mod record_array;
pub mod regular_array;
pub mod indexed_option_array;

use std::sync::Arc;

pub use numpy_array::NumpyArray;
pub use list_offset_array::ListOffsetArray;
pub use record_array::RecordArray;
pub use regular_array::RegularArray;
pub use indexed_option_array::IndexedOptionArray;

/// The core recursive content enum.
/// Every variant is heap-allocated and ref-counted via Arc for zero-copy slicing.
#[derive(Clone, Debug)]
pub enum Content {
    /// 1-D or rectilinear numeric data (float64 for now).
    NumpyArray(NumpyArray),

    /// Variable-length lists: offsets[i]..offsets[i+1] indexes into content.
    ListOffsetArray(ListOffsetArray),

    /// Fixed-length lists: list i is content[i*size .. (i+1)*size].
    RegularArray(RegularArray),

    /// Struct / record: parallel columns with optional field names.
    RecordArray(RecordArray),

    /// Optional (nullable) data: index[i] < 0 means None.
    IndexedOptionArray(IndexedOptionArray),

    /// Reserved for future use
    UnionArray(UnionArray),
}

// ── Inline stub types (not yet fully implemented) ────────────────────────────

#[derive(Clone, Debug)]
pub struct UnionArray {
    pub tags: Arc<[u8]>,
    pub index: Arc<[i64]>,
    pub contents: Vec<Arc<Content>>,
}

// ── Content methods ───────────────────────────────────────────────────────────

impl Content {
    /// Outer length (number of items at this level).
    pub fn len(&self) -> usize {
        match self {
            Content::NumpyArray(a) => a.len(),
            Content::ListOffsetArray(a) => a.len(),
            Content::RegularArray(a) => a.len(),
            Content::RecordArray(a) => a.len(),
            Content::IndexedOptionArray(a) => a.len(),
            Content::UnionArray(a) => a.tags.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Field access for RecordArray (and option-of-record, regular-of-record).
    pub fn get_field(&self, name: &str) -> Option<Arc<Content>> {
        match self {
            Content::RecordArray(r) => {
                let i = r.fields.iter().position(|f| f == name)?;
                Some(r.contents[i].clone())
            }
            Content::ListOffsetArray(a) => {
                // field access propagates through list structure
                let inner = a.content.get_field(name)?;
                Some(Arc::new(Content::ListOffsetArray(ListOffsetArray {
                    offsets: a.offsets.clone(),
                    content: inner,
                })))
            }
            Content::RegularArray(a) => {
                let inner = a.content.get_field(name)?;
                Some(Arc::new(Content::RegularArray(RegularArray {
                    content: inner,
                    size: a.size,
                    length: a.length,
                })))
            }
            Content::IndexedOptionArray(a) => {
                let inner = a.content.get_field(name)?;
                Some(Arc::new(Content::IndexedOptionArray(IndexedOptionArray {
                    index: a.index.clone(),
                    content: inner,
                })))
            }
            _ => None,
        }
    }

    /// Get field by position index (for tuple-style RecordArray with no names).
    pub fn get_field_at(&self, i: usize) -> Option<Arc<Content>> {
        match self {
            Content::RecordArray(r) => r.contents.get(i).cloned(),
            _ => None,
        }
    }
}

// ── Content traits ────────────────────────────────────────────────────────────

pub trait ArrayLike {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}

pub trait SequentialArray: ArrayLike {
    fn offsets(&self) -> &[i64];
    fn content(&self) -> &Content;
}

pub trait MappingArray: ArrayLike {
    fn fields(&self) -> &[String];
    fn field(&self, name: &str) -> Option<&Content>;
}

impl ArrayLike for ListOffsetArray {
    fn len(&self) -> usize { self.offsets.len().saturating_sub(1) }
}

impl ArrayLike for RecordArray {
    fn len(&self) -> usize { self.length }
}

impl SequentialArray for ListOffsetArray {
    fn offsets(&self) -> &[i64] { &self.offsets }
    fn content(&self) -> &Content { &self.content }
}

impl MappingArray for RecordArray {
    fn fields(&self) -> &[String] { &self.fields }
    fn field(&self, name: &str) -> Option<&Content> {
        self.fields.iter().position(|f| f == name)
            .map(|i| &*self.contents[i])
    }
}

// ── Merge helpers (used by convert.rs) ───────────────────────────────────────

/// Merge a homogeneous Vec<Content> of the same variant into a single Content.
/// Returns None if children is empty or types are mixed.
pub fn merge_contents_same_type(mut children: Vec<Content>) -> Option<Content> {
    if children.is_empty() {
        return Some(Content::NumpyArray(NumpyArray {
            data: Arc::from(vec![].into_boxed_slice()),
            shape: vec![0],
            strides: vec![1],
        }));
    }

    // All NumpyArray
    if children.iter().all(|c| matches!(c, Content::NumpyArray(_))) {
        let flat: Vec<f64> = children.iter().flat_map(|c| match c {
            Content::NumpyArray(a) => a.data.iter().copied().collect::<Vec<_>>(),
            _ => unreachable!(),
        }).collect();
        let len = flat.len();
        return Some(Content::NumpyArray(NumpyArray {
            data: Arc::from(flat.into_boxed_slice()),
            shape: vec![len],
            strides: vec![1],
        }));
    }

    // All RecordArray — merge column-wise
    if children.iter().all(|c| matches!(c, Content::RecordArray(_))) {
        let fields = match &children[0] {
            Content::RecordArray(r) => r.fields.clone(),
            _ => unreachable!(),
        };
        let n = children.len();
        let mut per_field: Vec<Vec<Content>> = vec![Vec::new(); fields.len()];
        for child in children.drain(..) {
            if let Content::RecordArray(r) = child {
                for (i, c) in r.contents.into_iter().enumerate() {
                    per_field[i].push(Arc::try_unwrap(c).unwrap_or_else(|arc| (*arc).clone()));
                }
            }
        }
        let contents = per_field.into_iter()
            .map(|items| Arc::new(merge_contents_same_type(items).unwrap()))
            .collect();
        return Some(Content::RecordArray(RecordArray { fields, contents, length: n }));
    }

    // All ListOffsetArray — merge into single flat ListOffsetArray
    if children.iter().all(|c| matches!(c, Content::ListOffsetArray(_))) {
        let mut merged_offsets: Vec<i64> = vec![0];
        let mut flat_values: Vec<f64> = Vec::new();
        for child in children.drain(..) {
            if let Content::ListOffsetArray(a) = child {
                let base = *merged_offsets.last().unwrap();
                for &o in a.offsets.iter().skip(1) {
                    merged_offsets.push(base + o);
                }
                if let Content::NumpyArray(inner) =
                    Arc::try_unwrap(a.content).unwrap_or_else(|arc| (*arc).clone())
                {
                    flat_values.extend_from_slice(&inner.data);
                }
            }
        }
        let len = flat_values.len();
        return Some(Content::ListOffsetArray(ListOffsetArray {
            offsets: Arc::from(merged_offsets.into_boxed_slice()),
            content: Arc::new(Content::NumpyArray(NumpyArray {
                data: Arc::from(flat_values.into_boxed_slice()),
                shape: vec![len],
                strides: vec![1],
            })),
        }));
    }

    None
}

// #[derive(Clone, Debug)]
// pub enum Content {
//     NumpyArray(NumpyArray),
//     ListOffsetArray(ListOffsetArray),
//     RegularArray(RegularArray),
//     RecordArray(RecordArray),
//     UnionArray(UnionArray),
// }

// impl Content {
//     pub fn len(&self) -> usize {
//         match self {
//             Content::NumpyArray(a) => a.len(),
//             Content::ListOffsetArray(a) => a.len(),
//             Content::RecordArray(a) => a.len(),
//             &Content::RegularArray(_) | &Content::UnionArray(_) => todo!(),
//         }
//     }

//     pub fn is_empty(&self) -> bool {
//         self.len() == 0
//     }

//     pub fn get_field(&self, i: usize) -> Option<&Arc<Content>> {
//         match self {
//             Content::RecordArray(r) => r.get_field(i),
//             _ => None,
//         }
//     }
// }

// #[derive(Clone, Debug)]
// pub struct RegularArray {
//     pub size: usize,           // fixed size per slot
//     pub content: Arc<Content>, // length = size * len(self)
// }

// #[derive(Clone, Debug)]
// pub struct UnionArray {
//     pub tags: Arc<[u8]>,       // which variant
//     pub index: Arc<[i64]>,     // index into each child
//     pub contents: Vec<Arc<Content>>,
// }

// pub trait ArrayLike {
//     fn len(&self) -> usize;
//     fn is_empty(&self) -> bool { self.len() == 0 }
// }

// pub trait SequentialArray: ArrayLike {
//     fn offsets(&self) -> &[i64];
//     fn content(&self) -> &Content;
// }

// pub trait MappingArray: ArrayLike {
//     fn fields(&self) -> &[String];
//     fn field(&self, name: &str) -> Option<&Content>;
// }

// impl ArrayLike for ListOffsetArray {
//     fn len(&self) -> usize {
//         self.offsets.len() - 1
//     }
// }

// impl ArrayLike for RecordArray {
//     fn len(&self) -> usize {
//         self.length
//     }
// }

// impl SequentialArray for ListOffsetArray {
//     fn offsets(&self) -> &[i64] { &self.offsets }
//     fn content(&self) -> &Content { &self.content }
// }

// impl MappingArray for RecordArray {
//     fn fields(&self) -> &[String] { &self.fields }
//     fn field(&self, name: &str) -> Option<&Content> {
//         self.fields.iter().position(|f| f == name)
//             .map(|i| &*self.contents[i])
//     }
// }
