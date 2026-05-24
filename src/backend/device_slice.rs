// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::ffi::c_void;
use std::marker::PhantomData;

/// Universal device slice type used by all backends.
/// Backends allocate memory differently, but all expose a raw pointer.
pub struct DevSlice<T> {
    pub ptr: *mut c_void,
    /// Passed to `free_fn` on drop. Equals `ptr` for most backends; may
    /// differ for Metal, where the allocation handle is a separate Box.
    free_data: *mut c_void,
    pub len: usize,
    free_fn: unsafe fn(*mut c_void),
    _marker: PhantomData<*mut T>,
}

/// DevSlice owns the memory. Requires the free function to be stored
/// alongside the pointer, since Drop can't know which backend allocated it.
impl<T> Drop for DevSlice<T> {
    fn drop(&mut self) {
        unsafe { (self.free_fn)(self.free_data) }
    }
}

/// Safety: the pointer is only accessed through the HIP/CUDA API,
/// which handles its own synchronisation.
unsafe impl<T: Send> Send for DevSlice<T> {}
unsafe impl<T: Sync> Sync for DevSlice<T> {}

impl<T> DevSlice<T> {
    /// Standard constructor. `free_fn` will be called with `ptr` on drop.
    #[inline]
    pub fn new(ptr: *mut c_void, len: usize, free_fn: unsafe fn(*mut c_void)) -> Self {
        Self {
            ptr,
            free_data: ptr,
            len,
            free_fn,
            _marker: PhantomData,
        }
    }

    /// Returns the raw allocation-handle pointer stored alongside this slice.
    ///
    /// For most backends this equals `self.ptr`.  For Metal it is the
    /// `Box<MetalAllocation>` raw pointer that keeps the `Buffer` alive.
    /// Only meaningful to the backend that created the slice; do not use
    /// outside `crate::backend`.
    #[cfg(target_os = "macos")]
    #[inline]
    pub(crate) fn free_data(&self) -> *mut c_void {
        self.free_data
    }

    /// Like `new`, but `free_fn` is called with `free_data` rather than `ptr`
    /// on drop. Use this when the allocation handle differs from the data
    /// pointer — e.g. Metal, where a `Box<MetalAllocation>` keeps the buffer
    /// alive while `ptr` is the CPU-visible contents pointer.
    #[inline]
    pub fn new_with_free_data(
        ptr: *mut c_void,
        len: usize,
        free_data: *mut c_void,
        free_fn: unsafe fn(*mut c_void),
    ) -> Self {
        Self {
            ptr,
            free_data,
            len,
            free_fn,
            _marker: PhantomData,
        }
    }
}

/// Trait for extracting a raw device pointer from any backend slice.
/// HIP, CUDA, CPU, etc. all implement this for DevSlice<T>.
pub trait DevicePtr {
    fn as_device_ptr(&self) -> *mut c_void;
}

impl<T> DevicePtr for DevSlice<T> {
    #[inline]
    fn as_device_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl<T> DevicePtr for &DevSlice<T> {
    fn as_device_ptr(&self) -> *mut c_void {
        self.ptr
    }
}
