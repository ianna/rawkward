// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::ffi::c_void;
use std::marker::PhantomData;

/// Universal device slice type used by all backends.
/// Backends allocate memory differently, but all expose a raw pointer.
pub struct DevSlice<T> {
    pub ptr: *mut c_void,
    pub len: usize,
    free_fn: unsafe fn(*mut c_void),
    _marker: PhantomData<*mut T>,
}

/// DevSlice owns the memory. Requires the free function to be stored
/// alongside the pointer, since Drop can't know which backend allocated it.
impl<T> Drop for DevSlice<T> {
    fn drop(&mut self) {
        unsafe { (self.free_fn)(self.ptr) }
    }
}

/// Safety: the pointer is only accessed through the HIP/CUDA API,
/// which handles its own synchronisation.
unsafe impl<T: Send> Send for DevSlice<T> {}
unsafe impl<T: Sync> Sync for DevSlice<T> {}

impl<T> DevSlice<T> {
    #[inline]
    pub fn new(ptr: *mut c_void, len: usize, free_fn: unsafe fn(*mut c_void)) -> Self {
        Self {
            ptr,
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
