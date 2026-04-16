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
                let mut new_offsets = vec![0i64];
                let mut flat: Vec<Content> = Vec::new();

                for i in start..stop {
                    if let Some(item) = inner.slice(i) {
                        new_offsets.push(new_offsets.last().unwrap() + 1);
                        flat.push(item);
                    }
                }

                let length = new_offsets.len() - 1;

                Some(Content::ListOffsetArray(ListOffsetArray {
                    offsets: Arc::from(new_offsets.into_boxed_slice()),
                    content: Arc::new(Content::RecordArray(super::RecordArray {
                        fields: vec![],
                        contents: flat.into_iter().map(Arc::new).collect(),
                        length,
                    })),
                }))
            }

            _ => unimplemented!("record slicing not implemented"),
        }
    }
}
