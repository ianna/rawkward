// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! CUDA CCCL/CUB backend (option B).
//!
//! `CudaComputeBackend` is a thin newtype over [`CudaBackend`]: it reuses the
//! Driver-API context, module, memory management and stream, and delegates the
//! whole [`GpuBackend`] contract to it. The difference is intent — code that
//! holds a `CudaComputeBackend` runs its reductions through the CUB
//! `DeviceSegmentedReduce` dispatch in `crate::kernels::cuda_compute::reduce`
//! (host-orchestrated CCCL, matching awkward-array's CUDA path) instead of the
//! hand-written kernels driven by a plain `CudaBackend`.
//!
//! Both share one device stream, so the CUB calls and any `cuLaunchKernel`
//! kernels order correctly against each other. This module is only compiled when
//! build.rs set `cfg(cuda_toolkit)`.

use std::ffi::c_void;

use crate::backend::cuda::CudaBackend;
use crate::backend::{DevSlice, GpuBackend, GpuError, GpuStream};

/// A CUDA backend whose reducers use CUB (`cuda.compute`-style), reusing all of
/// [`CudaBackend`]'s device resources.
pub struct CudaComputeBackend {
    inner: CudaBackend,
}

impl CudaComputeBackend {
    pub fn new() -> Result<Self, GpuError> {
        Ok(Self {
            inner: CudaBackend::new()?,
        })
    }

    /// Borrow the underlying driver-API backend (e.g. to launch a plain kernel
    /// on the same stream the CUB reducers use).
    #[inline]
    pub fn inner(&self) -> &CudaBackend {
        &self.inner
    }
}

impl GpuBackend for CudaComputeBackend {
    type DevSlice<T: Send + Sync> = DevSlice<T>;
    type KernelHandle = <CudaBackend as GpuBackend>::KernelHandle;

    unsafe fn alloc_slice<T: Copy + Send + Sync>(
        &self,
        len: usize,
    ) -> Result<Self::DevSlice<T>, GpuError> {
        unsafe { self.inner.alloc_slice(len) }
    }

    fn upload_slice<T: Copy + Send + Sync>(
        &self,
        host: &[T],
    ) -> Result<Self::DevSlice<T>, GpuError> {
        self.inner.upload_slice(host)
    }

    fn download_slice<T: Copy + Send + Sync>(
        &self,
        dev: &Self::DevSlice<T>,
        host: &mut [T],
    ) -> Result<(), GpuError> {
        self.inner.download_slice(dev, host)
    }

    fn get_kernel(&self, name: &str) -> Result<Self::KernelHandle, String> {
        self.inner.get_kernel(name)
    }

    unsafe fn launch(
        &self,
        kernel: &Self::KernelHandle,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[*mut c_void],
    ) -> Result<(), GpuError> {
        unsafe { self.inner.launch(kernel, grid, block, args) }
    }

    fn stream(&self) -> &GpuStream {
        self.inner.stream()
    }
}
