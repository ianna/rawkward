// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

use crate::dtype::DType;
use crate::layout::Layout;

pub mod indexed_option_array;
pub mod list_offset_array;
pub mod numpy_array;
pub mod record_array;
pub mod regular_array;

use std::sync::Arc;

pub use indexed_option_array::IndexedOptionArray;
pub use list_offset_array::ListOffsetArray;
pub use numpy_array::NumpyArray;
pub use record_array::RecordArray;
pub use regular_array::RegularArray;

/// The core recursive content enum.
/// Every variant is heap-allocated and ref-counted via Arc for zero-copy slicing.
#[derive(Clone, Debug)]
pub enum Content {
    /// 1-D or rectilinear numeric data.
    Bool(NumpyArray<bool>),
    I64(NumpyArray<i64>),
    F64(NumpyArray<f64>),

    /// Variable-length lists: offsets[i]..offsets[i+1] indexes into content.
    ListOffsetArray(ListOffsetArray),

    /// Fixed-length lists: list i is content[i*size .. (i+1)*size].
    RegularArray(RegularArray),

    /// Struct / record: parallel columns with optional field names.
    RecordArray(RecordArray),

    /// Optional (nullable) data: index[i] < 0 means None.
    IndexedOptionArray(IndexedOptionArray),

    /// Reserved for future use.
    UnionArray(UnionArray),
}

impl Layout for Content {
    fn len(&self) -> usize {
        match self {
            Content::Bool(a) => a.len(),
            Content::I64(a) => a.len(),
            Content::F64(a) => a.len(),
            Content::ListOffsetArray(a) => a.len(),
            Content::RegularArray(a) => a.len(),
            Content::RecordArray(a) => a.len(),
            Content::IndexedOptionArray(a) => a.len(),
            Content::UnionArray(_) => unimplemented!("UnionArray::len"),
        }
    }

    fn getitem(&self, index: usize) -> Arc<dyn Layout> {
        match self {
            Content::Bool(a) => a.getitem(index),
            Content::I64(a) => a.getitem(index),
            Content::F64(a) => a.getitem(index),
            Content::ListOffsetArray(a) => a.getitem(index),
            Content::RegularArray(a) => a.getitem(index),
            Content::RecordArray(a) => a.getitem(index),
            Content::IndexedOptionArray(a) => a.getitem(index),
            Content::UnionArray(_) => unimplemented!("UnionArray::getitem"),
        }
    }

    fn slice(&self, start: usize, stop: usize) -> Arc<dyn Layout> {
        match self {
            Content::Bool(a) => a.slice(start, stop),
            Content::I64(a) => a.slice(start, stop),
            Content::F64(a) => a.slice(start, stop),
            Content::ListOffsetArray(a) => a.slice(start, stop),
            Content::RegularArray(a) => a.slice(start, stop),
            Content::RecordArray(a) => a.slice(start, stop),
            Content::IndexedOptionArray(a) => a.slice(start, stop),
            Content::UnionArray(_) => unimplemented!("UnionArray::slice"),
        }
    }

    fn typetracer(&self) -> Arc<dyn Layout> {
        match self {
            Content::Bool(a) => a.typetracer(),
            Content::I64(a) => a.typetracer(),
            Content::F64(a) => a.typetracer(),
            Content::ListOffsetArray(a) => a.typetracer(),
            Content::RegularArray(_) => unimplemented!("RegularArray::typetracer"),
            Content::RecordArray(_) => unimplemented!("RecordArray::typetracer"),
            Content::IndexedOptionArray(_) => unimplemented!("IndexedOptionArray::typetracer"),
            Content::UnionArray(_) => unimplemented!("UnionArray::typetracer"),
        }
    }
}

// ── Inline stub types ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct UnionArray {
    pub tags: Arc<[u8]>,
    pub index: Arc<[i64]>,
    pub contents: Vec<Arc<Content>>,
}

// ── Content methods ───────────────────────────────────────────────────────────

impl Content {
    pub fn len(&self) -> usize {
        match self {
            Content::Bool(a) => a.len(),
            Content::I64(a) => a.len(),
            Content::F64(a) => a.len(),
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

    pub fn dtype(&self) -> Option<DType> {
        match self {
            Content::Bool(a) => Some(a.dtype),
            Content::I64(a) => Some(a.dtype),
            Content::F64(a) => Some(a.dtype),
            _ => None,
        }
    }

    /// Field access for RecordArray (and option-of-record, regular-of-record).
    pub fn get_field(&self, name: &str) -> Option<Arc<Content>> {
        match self {
            Content::RecordArray(r) => {
                let i = r.fields.iter().position(|f| f == name)?;
                Some(r.contents[i].clone()) // Arc::clone — free
            }
            Content::ListOffsetArray(a) => {
                let inner = a.content.get_field(name)?;
                Some(Arc::new(Content::ListOffsetArray(ListOffsetArray {
                    offsets: a.offsets.clone(), // Arc::clone — free
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
                    index: a.index.clone(), // Arc::clone — free
                    content: inner,
                })))
            }
            _ => None,
        }
    }

    /// Get field by position index (for tuple-style RecordArray with no names).
    pub fn get_field_at(&self, i: usize) -> Option<Arc<Content>> {
        match self {
            Content::RecordArray(r) => r.contents.get(i).cloned(), // Arc::clone — free
            Content::ListOffsetArray(a) => {
                let inner = a.content.get_field_at(i)?;
                Some(Arc::new(Content::ListOffsetArray(ListOffsetArray {
                    offsets: a.offsets.clone(),
                    content: inner,
                })))
            }
            Content::RegularArray(a) => {
                let inner = a.content.get_field_at(i)?;
                Some(Arc::new(Content::RegularArray(RegularArray {
                    content: inner,
                    size: a.size,
                    length: a.length,
                })))
            }
            Content::IndexedOptionArray(a) => {
                let inner = a.content.get_field_at(i)?;
                Some(Arc::new(Content::IndexedOptionArray(IndexedOptionArray {
                    index: a.index.clone(),
                    content: inner,
                })))
            }
            _ => None,
        }
    }

    /// Like Layout::slice but returns Arc<Content> for internal struct field use.
    pub fn slice_arc(&self, start: usize, stop: usize) -> Arc<Content> {
        match self {
            Content::Bool(a) => Arc::new(Content::Bool(a.slice_range(start, stop))),
            Content::I64(a) => Arc::new(Content::I64(a.slice_range(start, stop))),
            Content::F64(a) => Arc::new(Content::F64(a.slice_range(start, stop))),
            Content::ListOffsetArray(a) => {
                let base = a.offsets[start];
                let new_offsets: Arc<[i64]> = a.offsets[start..=stop]
                    .iter()
                    .map(|&o| o - base)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
                    .into();
                let cs = a.offsets[start] as usize;
                let ce = a.offsets[stop] as usize;
                Arc::new(Content::ListOffsetArray(ListOffsetArray {
                    offsets: new_offsets,
                    content: a.content.slice_arc(cs, ce),
                }))
            }
            Content::RecordArray(r) => {
                let contents = r
                    .contents
                    .iter()
                    .map(|c| c.slice_arc(start, stop))
                    .collect();
                Arc::new(Content::RecordArray(RecordArray {
                    fields: r.fields.clone(), // Arc<[Arc<str>]> clone once refactored
                    contents,
                    length: stop - start,
                }))
            }
            Content::IndexedOptionArray(a) => Arc::new(Content::IndexedOptionArray(
                crate::content::IndexedOptionArray {
                    index: Arc::from(a.index[start..stop].to_vec().into_boxed_slice()),
                    content: a.content.clone(),
                },
            )),
            _ => Arc::new(self.clone()),
        }
    }

    /// Like Layout::typetracer but returns Arc<Content>.
    pub fn typetracer_arc(&self) -> Arc<Content> {
        match self {
            Content::Bool(_) => Arc::new(Content::Bool(NumpyArray {
                data: Arc::from([] as [bool; 0]),
                shape: vec![0],
                strides: vec![1],
                dtype: DType::Bool,
            })),
            Content::I64(_) => Arc::new(Content::I64(NumpyArray {
                data: Arc::from([] as [i64; 0]),
                shape: vec![0],
                strides: vec![1],
                dtype: DType::I64,
            })),
            Content::F64(_) => Arc::new(Content::F64(NumpyArray {
                data: Arc::from([] as [f64; 0]),
                shape: vec![0],
                strides: vec![1],
                dtype: DType::F64,
            })),
            Content::ListOffsetArray(a) => Arc::new(Content::ListOffsetArray(ListOffsetArray {
                offsets: Arc::from([0_i64] as [i64; 1]),
                content: a.content.typetracer_arc(),
            })),
            _ => Arc::new(self.clone()),
        }
    }
}

// ── Merge helpers (used by convert.rs) ───────────────────────────────────────

/// Merge a homogeneous Vec<Content> of the same variant into a single Content.
/// Returns None if children is empty or types are mixed.
pub fn merge_contents_same_type(children: Vec<Content>) -> Option<Content> {
    if children.is_empty() {
        return Some(Content::F64(NumpyArray {
            data: Arc::from([] as [f64; 0]),
            shape: vec![0],
            strides: vec![1],
            dtype: DType::F64,
        }));
    }

    // ── Numeric variants: must copy data to produce a flat buffer ─────────────

    if children.iter().all(|c| matches!(c, Content::Bool(_))) {
        let flat: Vec<bool> = children
            .iter()
            .flat_map(|c| match c {
                Content::Bool(a) => a.data.iter().copied().collect::<Vec<_>>(),
                _ => unreachable!(),
            })
            .collect();
        let len = flat.len();
        return Some(Content::Bool(NumpyArray {
            data: Arc::from(flat.into_boxed_slice()),
            shape: vec![len],
            strides: vec![1],
            dtype: DType::Bool,
        }));
    }

    if children.iter().all(|c| matches!(c, Content::I64(_))) {
        let flat: Vec<i64> = children
            .iter()
            .flat_map(|c| match c {
                Content::I64(a) => a.data.iter().copied().collect::<Vec<_>>(),
                _ => unreachable!(),
            })
            .collect();
        let len = flat.len();
        return Some(Content::I64(NumpyArray {
            data: Arc::from(flat.into_boxed_slice()),
            shape: vec![len],
            strides: vec![1],
            dtype: DType::I64,
        }));
    }

    if children.iter().all(|c| matches!(c, Content::F64(_))) {
        let flat: Vec<f64> = children
            .iter()
            .flat_map(|c| match c {
                Content::F64(a) => a.data.iter().copied().collect::<Vec<_>>(),
                _ => unreachable!(),
            })
            .collect();
        let len = flat.len();
        return Some(Content::F64(NumpyArray {
            data: Arc::from(flat.into_boxed_slice()),
            shape: vec![len],
            strides: vec![1],
            dtype: DType::F64,
        }));
    }

    // ── RecordArray: merge column-wise ────────────────────────────────────────

    if children
        .iter()
        .all(|c| matches!(c, Content::RecordArray(_)))
    {
        let fields = match &children[0] {
            Content::RecordArray(r) => r.fields.clone(),
            _ => unreachable!(),
        };
        let total_rows: usize = children
            .iter()
            .map(|c| match c {
                Content::RecordArray(r) => r.length,
                _ => unreachable!(),
            })
            .sum();

        let num_fields = fields.len();
        // Collect per-column as Arc<Content> — no unwrap, no deep clone.
        let mut per_field: Vec<Vec<Arc<Content>>> = vec![Vec::new(); num_fields];
        for child in children {
            if let Content::RecordArray(r) = child {
                for (i, arc) in r.contents.into_iter().enumerate() {
                    per_field[i].push(arc); // Arc::clone implicit via move
                }
            }
        }
        let contents: Vec<Arc<Content>> = per_field
            .into_iter()
            .map(|arcs| Arc::new(merge_arc_contents(arcs).unwrap()))
            .collect();
        return Some(Content::RecordArray(RecordArray {
            fields,
            contents,
            length: total_rows,
        }));
    }

    // ── ListOffsetArray: concatenate sub-lists ────────────────────────────────
    if children
        .iter()
        .all(|c| matches!(c, Content::ListOffsetArray(_)))
    {
        let mut merged_offsets: Vec<i64> = vec![0];
        let mut inner_arcs: Vec<Arc<Content>> = Vec::new();

        for child in children {
            if let Content::ListOffsetArray(a) = child {
                let base = *merged_offsets.last().unwrap();
                for &o in a.offsets.iter().skip(1) {
                    merged_offsets.push(base + o);
                }
                inner_arcs.push(a.content); // Arc move — free
            }
        }
        let merged_inner = merge_arc_contents(inner_arcs)?;
        return Some(Content::ListOffsetArray(ListOffsetArray {
            offsets: Arc::from(merged_offsets.into_boxed_slice()),
            content: Arc::new(merged_inner),
        }));
    }

    None
}

/// Merge a Vec<Arc<Content>> of the same variant.
/// Accepts Arc<Content> directly — no unwrapping, no deep clones.
fn merge_arc_contents(arcs: Vec<Arc<Content>>) -> Option<Content> {
    // Unwrap each Arc only if we are the sole owner (common after drain);
    // otherwise clone the inner Content. Since these Arcs were just moved
    // out of their owning structs, try_unwrap succeeds in the common case.
    // When it fails (shared Arc), the clone is of a single array node,
    // not a deep subtree.
    let children: Vec<Content> = arcs
        .into_iter()
        .map(|arc| Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone()))
        .collect();
    merge_contents_same_type(children)
}
