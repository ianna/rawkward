// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::backend::DevicePtr;
use crate::backend::GpuStream;

pub trait GpuBackend {
    type DevSlice<T: Send + Sync>: DevicePtr + Send + Sync;
    type KernelHandle;

    /// Allocate an uninitialised device-side buffer of `len` elements.
    ///
    /// # Safety
    ///
    /// The returned slice contains uninitialised memory. Callers must write to
    /// every element before reading, or use the slice only as a write target
    /// (e.g. as the destination of a `download_slice` call).
    unsafe fn alloc_slice<T: Copy + Send + Sync>(&self, len: usize) -> Self::DevSlice<T>;
    fn upload_slice<T: Copy + Send + Sync>(&self, host: &[T]) -> Self::DevSlice<T>;
    fn download_slice<T: Copy + Send + Sync>(&self, dev: &Self::DevSlice<T>, host: &mut [T]);

    fn get_kernel(&self, name: &str) -> Result<Self::KernelHandle, String>;

    /// Launch a GPU kernel.
    ///
    /// # Safety
    ///
    /// * `kernel` must have been obtained from the same backend instance via
    ///   `get_kernel`.
    /// * `args` must be a valid set of pointers whose pointees match the
    ///   kernel's parameter types and sizes exactly.
    /// * Grid and block dimensions must be within the device's limits.
    unsafe fn launch(
        &self,
        kernel: &Self::KernelHandle,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[*mut std::ffi::c_void],
    );

    fn stream(&self) -> &GpuStream;
}
