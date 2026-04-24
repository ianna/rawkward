// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::dtype::{DType, ToDType};
use crate::layout::Layout;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct NumpyArray<T> {
    pub data: Arc<[T]>,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub dtype: DType,
}

impl<T: Copy + Send + Sync + 'static + std::fmt::Debug> Layout for NumpyArray<T> {
    fn len(&self) -> usize {
        self.shape[0]
    }

    fn getitem(&self, index: usize) -> Arc<dyn Layout> {
        // Zero-copy 1-element slice
        Arc::new(NumpyArray {
            data: Arc::from(&self.data[index..index + 1]),
            shape: vec![1],
            strides: vec![1],
            dtype: self.dtype,
        })
    }

    fn slice(&self, start: usize, stop: usize) -> Arc<dyn Layout> {
        // Zero-copy contiguous slice
        Arc::new(NumpyArray {
            data: Arc::from(&self.data[start..stop]),
            shape: vec![stop - start],
            strides: vec![1],
            dtype: self.dtype,
        })
    }

    fn typetracer(&self) -> Arc<dyn Layout> {
        // Empty data, same dtype, shape = [0]
        Arc::new(NumpyArray {
            data: Arc::from([] as [T; 0]),
            shape: vec![0],
            strides: vec![1],
            dtype: self.dtype,
        })
    }
}

impl<T: Copy + Send + Sync + 'static + ToDType + Default> NumpyArray<T> {
    pub fn from_vec(v: Vec<T>) -> Self
    where
        T: Copy + Send + Sync + 'static + ToDType,
    {
        let len = v.len();
        let dtype = T::default().dtype();
        NumpyArray {
            data: Arc::from(v.into_boxed_slice()),
            shape: vec![len],
            strides: vec![1],
            dtype,
        }
    }
    pub fn len(&self) -> usize {
        self.shape[0]
    }

    pub fn is_empty(&self) -> bool {
        self.shape[0] == 0
    }

    /// Zero-copy sub-slice returning a new NumpyArray sharing the same Arc buffer.
    pub fn slice_range(&self, start: usize, stop: usize) -> Self {
        NumpyArray {
            data: Arc::from(&self.data[start..stop]),
            shape: vec![stop - start],
            strides: vec![1],
            dtype: self.dtype,
        }
    }
}
