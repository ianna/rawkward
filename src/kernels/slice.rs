// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::sync::Arc;
use crate::content::{Content, NumpyArray, ListOffsetArray, RecordArray};

#[derive(Debug)]
pub enum Slice {
    Index(i64),
}

#[derive(thiserror::Error, Debug)]
pub enum SliceError {
    #[error("index out of bounds")]
    OutOfBounds,
}

pub fn slice(content: &Content, s: &Slice) -> Result<Content, SliceError> {
    match (content, s) {
        (Content::NumpyArray(a), Slice::Index(i)) => {
            let idx = normalize_index(*i, a.data.len())?;
            Ok(Content::NumpyArray(NumpyArray {
                data: Arc::from(&a.data[idx..idx + 1]),
                shape: vec![1],
                strides: vec![1],
            }))
        }
        (Content::ListOffsetArray(a), Slice::Index(i)) => {
            let idx = normalize_index(*i, a.len())?;
            a.slice(idx).ok_or(SliceError::OutOfBounds)
        }
        _ => unimplemented!("slice not implemented for this layout"),
    }
}

pub fn slice_range(content: &Content, start: usize, stop: usize) -> Result<Content, SliceError> {
    match content {
        Content::NumpyArray(a) => {
            if stop > a.data.len() { return Err(SliceError::OutOfBounds); }
            let data: Arc<[f64]> = Arc::from(&a.data[start..stop]);
            let len = data.len();
            Ok(Content::NumpyArray(NumpyArray {
                data,
                shape: vec![len],
                strides: vec![1],
            }))
        }
        Content::ListOffsetArray(a) => {
            if stop > a.len() { return Err(SliceError::OutOfBounds); }
            let new_base = a.offsets[start];
            let new_offsets: Arc<[i64]> = Arc::from(
                a.offsets[start..=stop].iter().map(|o| o - new_base).collect::<Vec<_>>().as_slice()
            );
            Ok(Content::ListOffsetArray(crate::content::ListOffsetArray {
                offsets: new_offsets,
                content: a.content.clone(),
            }))
        }
        _ => unimplemented!("slice_range not implemented for this layout"),
    }
}

fn normalize_index(i: i64, len: usize) -> Result<usize, SliceError> {
    let idx = if i < 0 { len as i64 + i } else { i } as usize;
    if idx >= len { Err(SliceError::OutOfBounds) } else { Ok(idx) }
}
