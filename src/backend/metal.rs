// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::ffi::c_void;

use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

use crate::backend::{DevSlice, GpuBackend, GpuError, GpuStream};
use crate::kernels::metal::{MetalKernelHandle, MetalKernelRegistry};

// ---------------------------------------------------------------------------
// Allocation lifetime management
// ---------------------------------------------------------------------------

/// Owns a Metal `Buffer`, keeping it alive for as long as the `DevSlice` that
/// points into it exists.  With `StorageModeShared`, the Metal runtime owns the
/// underlying memory; releasing the `Buffer` frees it.
///
/// Stored on the heap and accessed via a raw pointer that is threaded through
/// `DevSlice::free_data` so that `metal_free` can recover and drop it on
/// `DevSlice::drop`.
struct MetalAllocation {
    _buffer: Buffer,
}

/// Called by `DevSlice::drop` via the `free_data` field.
///
/// # Safety
///
/// `free_data` must be a pointer previously produced by
/// `Box::into_raw(Box::new(MetalAllocation { … }))`.  It must not be called
/// more than once for the same pointer.
unsafe fn metal_free(free_data: *mut c_void) {
    if !free_data.is_null() {
        // SAFETY: contract above.  Rust 2024 requires an explicit unsafe block
        // even inside an unsafe fn.
        unsafe { drop(Box::from_raw(free_data as *mut MetalAllocation)) };
    }
}

// ---------------------------------------------------------------------------
// MetalBackend
// ---------------------------------------------------------------------------

pub struct MetalBackend {
    pub device: Device,
    /// Retained so callers can enqueue work on the same queue as the stream.
    pub queue: CommandQueue,
    pub stream: GpuStream,
    /// Compiled shader library + pipeline cache.
    pub registry: MetalKernelRegistry,
}

/// Recover the underlying `metal::Buffer` from a `DevSlice` that was created
/// by `MetalBackend::alloc_slice` or `MetalBackend::upload_slice`.
///
/// # Safety
///
/// `slice` **must** have been created by a `MetalBackend` method.  Passing a
/// `DevSlice` from any other backend (HIP, CPU, …) is undefined behaviour
/// because `free_data` will not point to a `MetalAllocation`.
///
/// The returned reference borrows from the allocation stored in `slice`, so
/// it must not outlive `slice`.
pub(crate) unsafe fn buffer_of<T>(slice: &DevSlice<T>) -> &Buffer {
    // SAFETY: `free_data()` is `Box::into_raw(Box<MetalAllocation>)` for all
    // slices produced by this backend.  The Box is alive for the lifetime of
    // `slice`, so dereferencing is safe for at most that lifetime.
    unsafe {
        let alloc = &*(slice.free_data() as *const MetalAllocation);
        &alloc._buffer
    }
}

impl MetalBackend {
    pub fn new() -> Result<Self, GpuError> {
        let device = Device::system_default().ok_or(GpuError::InitializationFailed)?;
        let queue = device.new_command_queue();
        let registry = MetalKernelRegistry::new(&device).map_err(GpuError::MetalError)?;

        Ok(Self {
            device,
            // `queue` is an Obj-C reference-counted handle; clone bumps the
            // retain count so both `self.queue` and `self.stream` are valid.
            queue: queue.clone(),
            stream: GpuStream::Metal(queue),
            registry,
        })
    }
}

// ---------------------------------------------------------------------------
// GpuBackend impl
// ---------------------------------------------------------------------------

impl GpuBackend for MetalBackend {
    type DevSlice<T: Send + Sync> = DevSlice<T>;
    type KernelHandle = MetalKernelHandle;

    unsafe fn alloc_slice<T: Copy + Send + Sync>(&self, len: usize) -> Self::DevSlice<T> {
        let bytes = len * std::mem::size_of::<T>();

        // Metal's newBufferWithLength:0 returns a Buffer whose underlying
        // Objective-C object has a null internal pointer; dropping it panics in
        // metal-0.29.  Return an empty DevSlice with no Metal allocation instead.
        if bytes == 0 {
            return DevSlice::new_with_free_data(
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                metal_free, // metal_free is a no-op for null free_data
            );
        }

        let buffer = self
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);

        // `buffer.contents()` is the CPU-visible pointer into unified memory.
        let payload_ptr = buffer.contents() as *mut c_void;

        // Box the allocation tracker so the Metal buffer stays alive until the
        // DevSlice is dropped, at which point `metal_free` runs.
        let free_data = Box::into_raw(Box::new(MetalAllocation { _buffer: buffer })) as *mut c_void;

        DevSlice::new_with_free_data(payload_ptr, len, free_data, metal_free)
    }

    fn upload_slice<T: Copy + Send + Sync>(&self, host: &[T]) -> Self::DevSlice<T> {
        let bytes = host.len() * std::mem::size_of::<T>();

        // Same zero-length guard as alloc_slice: Metal returns a null-backed
        // Buffer for a 0-byte request, which panics on drop in metal-0.29.
        if bytes == 0 {
            return DevSlice::new_with_free_data(
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                metal_free,
            );
        }

        // metal 0.29 has no `new_buffer_with_bytes` (copying variant).  Allocate
        // a shared-mode buffer then memcpy the host data in — valid because
        // StorageModeShared exposes the same physical pages to both CPU and GPU.
        let buffer = self
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
        let payload_ptr = buffer.contents() as *mut c_void;
        unsafe {
            std::ptr::copy_nonoverlapping(
                host.as_ptr() as *const u8,
                payload_ptr as *mut u8,
                bytes,
            );
        }

        let free_data = Box::into_raw(Box::new(MetalAllocation { _buffer: buffer })) as *mut c_void;

        DevSlice::new_with_free_data(payload_ptr, host.len(), free_data, metal_free)
    }

    fn download_slice<T: Copy + Send + Sync>(&self, dev: &Self::DevSlice<T>, host: &mut [T]) {
        // With StorageModeShared the GPU and CPU share the same physical pages;
        // no explicit blit / DMA is needed — a plain memcpy suffices.
        // Guard against a shorter source so we never read past `dev.len`.
        let copy_len = host.len().min(dev.len);
        // copy_nonoverlapping requires both pointers to be non-null even for a
        // zero-length copy (Rust reference: "even if count is 0, the pointers
        // must be valid").  Zero-length DevSlices carry a null ptr, so skip.
        if copy_len == 0 {
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(dev.ptr as *const T, host.as_mut_ptr(), copy_len);
        }
    }

    fn get_kernel(&self, name: &str) -> Result<Self::KernelHandle, String> {
        self.registry
            .get_pipeline(&self.device, name)
            .map(MetalKernelHandle)
    }

    unsafe fn launch(
        &self,
        _kernel: &Self::KernelHandle,
        _grid: (u32, u32, u32),
        _block: (u32, u32, u32),
        _args: &[*mut c_void],
    ) {
        // TODO: Metal dispatch via ComputeCommandEncoder.
        //
        // Metal binds arguments as MTLBuffers or inline bytes rather than raw
        // pointer arrays, so the *mut c_void convention used by HIP/CUDA cannot
        // map directly.  A buffer registry or argument-reflection layer is
        // needed before this can be implemented:
        //
        //   let cmd_buf  = self.queue.new_command_buffer();
        //   let encoder  = cmd_buf.new_compute_command_encoder();
        //   encoder.set_compute_pipeline_state(&kernel.0);
        //   // for each arg: encoder.set_buffer(index, Some(&buf), 0)
        //   encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        //   encoder.end_encoding();
        //   cmd_buf.commit();
        //   cmd_buf.wait_until_completed();
        todo!("Metal kernel launch: implement buffer binding via ComputeCommandEncoder")
    }

    fn stream(&self) -> &GpuStream {
        &self.stream
    }
}
