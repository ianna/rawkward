// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::backend::DevicePtr;
use crate::backend::GpuStream;

pub trait GpuBackend {
    type DevSlice<T: Send + Sync>: DevicePtr + Send + Sync;
    type KernelHandle;

    unsafe fn alloc_slice<T: Copy + Send + Sync>(&self, len: usize) -> Self::DevSlice<T>;
    fn upload_slice<T: Copy + Send + Sync>(&self, host: &[T]) -> Self::DevSlice<T>;
    fn download_slice<T: Copy + Send + Sync>(&self, dev: &Self::DevSlice<T>, host: &mut [T]);

    fn get_kernel(&self, name: &str) -> Result<Self::KernelHandle, String>;
    unsafe fn launch(
        &self,
        kernel: &Self::KernelHandle,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[*mut std::ffi::c_void],
    );

    fn stream(&self) -> &GpuStream;
}
