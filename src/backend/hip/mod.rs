// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

pub mod hip_bindings {
    include!(env!("RAWKWARD_HIP_BINDINGS"));
    pub use hip_bindings::*;
}

use std::ffi::c_void;
use std::ptr;

use crate::backend::{DevSlice, GpuBackend, GpuStream};

use hip_bindings::{
    hipError_t_hipSuccess, hipFree, hipFunction_t, hipMalloc, hipMemcpy,
    hipMemcpyKind_hipMemcpyDeviceToHost, hipMemcpyKind_hipMemcpyHostToDevice, hipModule_t,
    hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoadData, hipModuleUnload, hipStream_t,
    hipStreamCreate, hipStreamDestroy,
};

fn hip_free(ptr: *mut c_void) {
    unsafe {
        hipFree(ptr);
    }
}

/// Resolved kernel handle — wraps a `hipFunction_t` obtained via `hipModuleGetFunction`.
pub struct HipKernelHandle(hipFunction_t);

pub struct HipBackend {
    module: hipModule_t,
    stream: GpuStream,
}

#[cfg(feature = "hip")]
static HIP_MODULE_BYTES: &[u8] = include_bytes!(env!("RAWKWARD_HIP_MODULE_PATH"));

impl HipBackend {
    pub fn new() -> Self {
        unsafe {
            let mut module: hipModule_t = std::mem::zeroed();
            let res = hipModuleLoadData(&mut module, HIP_MODULE_BYTES.as_ptr() as *const _);
            if res != hipError_t_hipSuccess {
                panic!("hipModuleLoadData failed: {}", res);
            }

            let mut raw_stream: hipStream_t = std::mem::zeroed();
            let res = hipStreamCreate(&mut raw_stream);
            if res != hipError_t_hipSuccess {
                panic!("hipStreamCreate failed: {}", res);
            }

            Self {
                module,
                stream: GpuStream::Hip(raw_stream),
            }
        }
    }
}

impl Drop for HipBackend {
    fn drop(&mut self) {
        unsafe {
            if let GpuStream::Hip(s) = self.stream {
                hipStreamDestroy(s);
            }
            hipModuleUnload(self.module);
        }
    }
}

impl GpuBackend for HipBackend {
    type DevSlice<T: Send + Sync> = DevSlice<T>;
    type KernelHandle = HipKernelHandle;

    unsafe fn alloc_slice<T: Copy + Send + Sync>(&self, len: usize) -> Self::DevSlice<T> {
        let bytes = len * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = ptr::null_mut();

        unsafe {
            let res = hipMalloc(&mut ptr, bytes);
            if res != hipError_t_hipSuccess {
                panic!("hipMalloc failed: {}", res);
            }
        }

        DevSlice::new(ptr, len, hip_free)
    }

    fn upload_slice<T: Copy + Send + Sync>(&self, host: &[T]) -> Self::DevSlice<T> {
        let bytes = host.len() * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = ptr::null_mut();

        unsafe {
            let res = hipMalloc(&mut ptr, bytes);
            if res != hipError_t_hipSuccess {
                panic!("hipMalloc failed: {}", res);
            }

            let res = hipMemcpy(
                ptr,
                host.as_ptr() as *const c_void,
                bytes,
                hipMemcpyKind_hipMemcpyHostToDevice,
            );
            if res != hipError_t_hipSuccess {
                hipFree(ptr);
                panic!("hipMemcpy HostToDevice failed: {}", res);
            }
        }

        DevSlice::new(ptr, host.len(), hip_free)
    }

    fn download_slice<T: Copy + Send + Sync>(&self, dev: &Self::DevSlice<T>, host: &mut [T]) {
        assert!(
            host.len() <= dev.len,
            "download_slice: host buffer ({} elems) exceeds device slice ({} elems) — would read out of bounds",
            host.len(),
            dev.len
        );
        let bytes = host.len() * std::mem::size_of::<T>();

        unsafe {
            let res = hipMemcpy(
                host.as_mut_ptr() as *mut c_void,
                dev.ptr,
                bytes,
                hipMemcpyKind_hipMemcpyDeviceToHost,
            );
            if res != hipError_t_hipSuccess {
                panic!("hipMemcpy DeviceToHost failed: {}", res);
            }
        }
    }

    fn get_kernel(&self, name: &str) -> Result<Self::KernelHandle, String> {
        unsafe {
            let mut func: hipFunction_t = std::mem::zeroed();
            let cname =
                std::ffi::CString::new(name).map_err(|e| format!("invalid kernel name: {e}"))?;

            let res = hipModuleGetFunction(&mut func, self.module, cname.as_ptr());
            if res != hipError_t_hipSuccess {
                return Err(format!("hipModuleGetFunction failed for {name:?}: {res}"));
            }

            Ok(HipKernelHandle(func))
        }
    }

    unsafe fn launch(
        &self,
        kernel: &Self::KernelHandle,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[*mut c_void],
    ) {
        let stream = match self.stream {
            GpuStream::Hip(s) => s,
            _ => panic!("HIP backend used with non-HIP stream"),
        };

        unsafe {
            let res = hipModuleLaunchKernel(
                kernel.0,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                stream,
                args.as_ptr() as *mut *mut c_void,
                ptr::null_mut(),
            );

            if res != hipError_t_hipSuccess {
                panic!("hipModuleLaunchKernel failed: {}", res);
            }
        }
    }

    fn stream(&self) -> &GpuStream {
        &self.stream
    }
}
