// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::sync::Arc;

use super::{Content, NumpyArray};

#[derive(Clone, Debug)]
pub struct ListOffsetArray {
    pub offsets: Arc<[i64]>,
    pub content: Arc<Content>,
}

impl ListOffsetArray {
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn slice(&self, idx: usize) -> Option<Content> {
        if idx + 1 >= self.offsets.len() {
            return None;
        }

        let start = self.offsets[idx] as usize;
        let stop = self.offsets[idx + 1] as usize;

        match self.content.as_ref() {
            Content::NumpyArray(inner) => {
                let data: Arc<[f64]> = Arc::from(&inner.data[start..stop]);
                let len = data.len();
                Some(Content::NumpyArray(NumpyArray {
                    data,
                    shape: vec![len],
                    strides: vec![1],
                }))
            }

            Content::ListOffsetArray(inner) => {
                // Return a new ListOffsetArray covering offsets[start..=stop]
                let new_base = inner.offsets[start];
                let new_offsets: Arc<[i64]> = Arc::from(
                    inner.offsets[start..=stop]
                        .iter()
                        .map(|o| o - new_base)
                        .collect::<Vec<_>>()
                        .as_slice(),
                );
                Some(Content::ListOffsetArray(ListOffsetArray {
                    offsets: new_offsets,
                    content: inner.content.clone(),
                }))
            }
            // Content::RecordArray(inner) => {
            //     let contents: Vec<Arc<Content>> = inner.contents.iter()
            //         .map(|col| Arc::new(slice_content(col, start, stop)))
            //         .collect();
            //     Some(Content::RecordArray(super::RecordArray {
            //         fields: inner.fields.clone(),
            //         contents,
            //         length: stop - start,
            //     }))
            // }
            Content::RecordArray(inner) => {
                // Slice each field's content
                let contents: Vec<Arc<Content>> = inner
                    .contents
                    .iter()
                    .map(|c| Arc::new(slice_content(c, start, stop)))
                    .collect();
                Some(Content::RecordArray(super::RecordArray {
                    fields: inner.fields.clone(),
                    contents,
                    length: stop - start,
                }))
            }

            _ => unimplemented!("slice not implemented for this layout"),
        }
    }
}

fn slice_content(c: &Content, start: usize, stop: usize) -> Content {
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
            let new_offsets: Arc<[i64]> = Arc::from(
                a.offsets[start..=stop]
                    .iter()
                    .map(|o| o - new_base)
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            Content::ListOffsetArray(ListOffsetArray {
                offsets: new_offsets,
                content: a.content.clone(),
            })
        }
        _ => unimplemented!("slice_content not implemented for this layout"),
    }
}
