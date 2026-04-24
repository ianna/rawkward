// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::content::{Content, IndexedOptionArray, NumpyArray, RecordArray};
use crate::dtype::DType;
use std::sync::Arc;

#[derive(Debug)]
pub enum Slice {
    Index(i64),
}

#[derive(thiserror::Error, Debug)]
pub enum SliceError {
    #[error("index out of bounds")]
    OutOfBounds,
    #[error("slice step must be 1")]
    StepNotOne,
}

/// Single-element index slice.
pub fn slice(content: &Content, s: &Slice) -> Result<Content, SliceError> {
    match (content, s) {
        (Content::Bool(a), Slice::Index(i)) => {
            let idx = normalize_index(*i, a.data.len())?;
            Ok(Content::Bool(NumpyArray {
                data: Arc::from(&a.data[idx..idx + 1]),
                shape: vec![1],
                strides: vec![1],
                dtype: DType::Bool,
            }))
        }

        (Content::I64(a), Slice::Index(i)) => {
            let idx = normalize_index(*i, a.data.len())?;
            Ok(Content::I64(NumpyArray {
                data: Arc::from(&a.data[idx..idx + 1]),
                shape: vec![1],
                strides: vec![1],
                dtype: DType::I64,
            }))
        }

        (Content::F64(a), Slice::Index(i)) => {
            let idx = normalize_index(*i, a.data.len())?;
            Ok(Content::F64(NumpyArray {
                data: Arc::from(&a.data[idx..idx + 1]),
                shape: vec![1],
                strides: vec![1],
                dtype: DType::F64,
            }))
        }

        (Content::ListOffsetArray(a), Slice::Index(i)) => {
            let idx = normalize_index(*i, a.len())?;
            let start = a.offsets[idx] as usize;
            let stop = a.offsets[idx + 1] as usize;
            Ok((*a.content.slice_arc(start, stop)).clone())
        }

        (Content::RegularArray(a), Slice::Index(i)) => {
            let idx = normalize_index(*i, a.length)?;
            let start = idx * a.size;
            let stop = start + a.size;
            Ok((*a.content.slice_arc(start, stop)).clone())
        }

        (Content::RecordArray(r), Slice::Index(i)) => {
            let idx = normalize_index(*i, r.length)?;
            let contents: Vec<Arc<Content>> = r
                .contents
                .iter()
                .map(|col| col.slice_arc(idx, idx + 1))
                .collect();
            Ok(Content::RecordArray(RecordArray {
                fields: r.fields.clone(),
                contents,
                length: 1,
            }))
        }

        (Content::IndexedOptionArray(a), Slice::Index(i)) => {
            let idx = normalize_index(*i, a.len())?;
            let index_val = a.index[idx];
            if index_val < 0 {
                // None: return length-1 IndexedOptionArray with index = -1
                Ok(Content::IndexedOptionArray(IndexedOptionArray {
                    index: Arc::from([-1i64] as [i64; 1]),
                    content: a.content.clone(), // Arc::clone — free
                }))
            } else {
                let pos = index_val as usize;
                Ok((*a.content.slice_arc(pos, pos + 1)).clone())
            }
        }

        _ => Err(SliceError::OutOfBounds),
    }
}

/// Range slice: content[start..stop].
pub fn slice_range(content: &Content, start: usize, stop: usize) -> Result<Content, SliceError> {
    if stop > content.len() {
        return Err(SliceError::OutOfBounds);
    }
    // slice_arc returns Arc<Content>; deref and clone to get Content
    Ok((*content.slice_arc(start, stop)).clone())
}

fn normalize_index(i: i64, len: usize) -> Result<usize, SliceError> {
    let idx = if i < 0 { len as i64 + i } else { i };
    if idx < 0 || idx as usize >= len {
        Err(SliceError::OutOfBounds)
    } else {
        Ok(idx as usize)
    }
}
