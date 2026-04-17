// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

pub mod numpy_array;
pub mod list_offset_array;
pub mod record_array;

use std::sync::Arc;

pub use numpy_array::NumpyArray;
pub use list_offset_array::ListOffsetArray;
pub use record_array::RecordArray;

#[derive(Clone, Debug)]
pub enum Content {
    NumpyArray(NumpyArray),
    ListOffsetArray(ListOffsetArray),
    RegularArray(RegularArray),
    RecordArray(RecordArray),
    UnionArray(UnionArray),
}

impl Content {
    pub fn len(&self) -> usize {
        match self {
            Content::NumpyArray(a) => a.len(),
            Content::ListOffsetArray(a) => a.len(),
            Content::RecordArray(a) => a.len(),
            &Content::RegularArray(_) | &Content::UnionArray(_) => todo!(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_field(&self, i: usize) -> Option<&Arc<Content>> {
        match self {
            Content::RecordArray(r) => r.get_field(i),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RegularArray {
    pub size: usize,           // fixed size per slot
    pub content: Arc<Content>, // length = size * len(self)
}

#[derive(Clone, Debug)]
pub struct UnionArray {
    pub tags: Arc<[u8]>,       // which variant
    pub index: Arc<[i64]>,     // index into each child
    pub contents: Vec<Arc<Content>>,
}

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
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }
}

impl ArrayLike for RecordArray {
    fn len(&self) -> usize {
        self.length
    }
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
