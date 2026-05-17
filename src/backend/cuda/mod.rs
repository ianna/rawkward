// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::backend::GpuStream;
use crate::backend::{DevSlice, GpuBackend, GpuError};

pub struct CudaBackend {
    stream: GpuStream,
}

/// Placeholder — CUDA is not yet implemented.
/// Satisfies the KernelHandle associated type requirement.
pub struct CudaKernelHandle;

impl CudaBackend {
    pub fn new() -> Result<Self, GpuError> {
        Err(GpuError::CudaError("CUDA backend not implemented".into()))
    }
}

impl GpuBackend for CudaBackend {
    type DevSlice<T: Send + Sync> = DevSlice<T>;
    type KernelHandle = CudaKernelHandle;

    unsafe fn alloc_slice<T: Copy + Send + Sync>(&self, _: usize) -> Self::DevSlice<T> {
        unimplemented!("CUDA backend not implemented")
    }

    fn upload_slice<T: Copy + Send + Sync>(&self, _: &[T]) -> Self::DevSlice<T> {
        unimplemented!("CUDA backend not implemented")
    }

    fn download_slice<T: Copy + Send + Sync>(&self, _: &Self::DevSlice<T>, _: &mut [T]) {
        unimplemented!("CUDA backend not implemented")
    }

    fn get_kernel(&self, name: &str) -> Result<Self::KernelHandle, String> {
        Err(format!(
            "CUDA backend not implemented (requested kernel: {name})"
        ))
    }

    unsafe fn launch(
        &self,
        _: &Self::KernelHandle,
        _: (u32, u32, u32),
        _: (u32, u32, u32),
        _: &[*mut std::ffi::c_void],
    ) {
        unimplemented!("CUDA backend not implemented")
    }

    fn stream(&self) -> &GpuStream {
        &self.stream
    }
}
