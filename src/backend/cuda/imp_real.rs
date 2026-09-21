// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Real CUDA backend via the CUDA Driver API. Compiled only when `build.rs`
//! set `cfg(cuda_toolkit)` (i.e. nvcc + the driver were found and the kernels
//! were built into a fatbin and bindings generated).
//!
//! This mirrors `backend/hip/mod.rs`: load a device module, look up kernels by
//! name, and launch them — but through `cu*` Driver-API calls instead of the
//! HIP runtime.
//!
//! NOTE (build-host verification): this file could not be compiled in the
//! development environment (no CUDA toolkit / GPU). The Driver-API symbol
//! spellings below follow bindgen's constified-enum convention (matching how
//! the HIP bindings expose `hipError_t_hipSuccess`), i.e.
//! `cudaError_enum_CUDA_SUCCESS` and the `_v2` versioned entry points. If the
//! generated `cuda_bindings.rs` names differ on your CUDA version, adjust the
//! `use` list and `ok()` accordingly — the control flow is otherwise complete.

pub mod cuda_bindings {
    include!(env!("RAWKWARD_CUDA_BINDINGS"));
    pub use cuda_bindings::*;
}

use std::ffi::{CString, c_void};
use std::ptr;

use crate::backend::{DevSlice, GpuBackend, GpuError, GpuStream};

use cuda_bindings::{
    CUcontext, CUdevice, CUdeviceptr, CUfunction, CUmodule, CUresult, CUstream, cuCtxCreate_v2,
    cuCtxDestroy_v2, cuDeviceGet, cuInit, cuLaunchKernel, cuMemAlloc_v2, cuMemFree_v2,
    cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, cuModuleGetFunction, cuModuleLoadData, cuModuleUnload,
    cuStreamCreate, cuStreamDestroy_v2,
};

/// The device module (fatbin) built from `src/kernels/cuda/**/*.cu` by build.rs.
static CUDA_MODULE_BYTES: &[u8] = include_bytes!(env!("RAWKWARD_CUDA_MODULE_PATH"));

#[inline]
fn ok(res: CUresult) -> bool {
    res == cuda_bindings::cudaError_enum_CUDA_SUCCESS
}

/// `free_fn` stored in each `DevSlice`; frees the device allocation.
fn cuda_free(ptr: *mut c_void) {
    unsafe {
        cuMemFree_v2(ptr as CUdeviceptr);
    }
}

/// Resolved kernel handle — wraps a `CUfunction` from `cuModuleGetFunction`.
pub struct CudaKernelHandle(CUfunction);

pub struct CudaBackend {
    context: CUcontext,
    module: CUmodule,
    stream: GpuStream,
}

impl CudaBackend {
    pub fn new() -> Result<Self, GpuError> {
        unsafe {
            let res = cuInit(0);
            if !ok(res) {
                return Err(GpuError::CudaError(format!("cuInit failed: {res:?}")));
            }

            let mut device: CUdevice = 0;
            let res = cuDeviceGet(&mut device, 0);
            if !ok(res) {
                return Err(GpuError::CudaError(format!("cuDeviceGet failed: {res:?}")));
            }

            // Create (and make current) a context for device 0.
            let mut context: CUcontext = ptr::null_mut();
            let res = cuCtxCreate_v2(&mut context, 0, device);
            if !ok(res) {
                return Err(GpuError::CudaError(format!("cuCtxCreate failed: {res:?}")));
            }

            let mut module: CUmodule = ptr::null_mut();
            let res = cuModuleLoadData(&mut module, CUDA_MODULE_BYTES.as_ptr() as *const c_void);
            if !ok(res) {
                cuCtxDestroy_v2(context);
                return Err(GpuError::CudaError(format!(
                    "cuModuleLoadData failed: {res:?}"
                )));
            }

            let mut raw_stream: CUstream = ptr::null_mut();
            let res = cuStreamCreate(&mut raw_stream, 0);
            if !ok(res) {
                cuModuleUnload(module);
                cuCtxDestroy_v2(context);
                return Err(GpuError::CudaError(format!(
                    "cuStreamCreate failed: {res:?}"
                )));
            }

            Ok(Self {
                context,
                module,
                stream: GpuStream::Cuda(raw_stream),
            })
        }
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        unsafe {
            if let GpuStream::Cuda(s) = self.stream {
                cuStreamDestroy_v2(s);
            }
            cuModuleUnload(self.module);
            cuCtxDestroy_v2(self.context);
        }
    }
}

impl GpuBackend for CudaBackend {
    type DevSlice<T: Send + Sync> = DevSlice<T>;
    type KernelHandle = CudaKernelHandle;

    unsafe fn alloc_slice<T: Copy + Send + Sync>(
        &self,
        len: usize,
    ) -> Result<Self::DevSlice<T>, GpuError> {
        let bytes = len * std::mem::size_of::<T>();
        let mut dptr: CUdeviceptr = 0;

        unsafe {
            let res = cuMemAlloc_v2(&mut dptr, bytes);
            if !ok(res) {
                return Err(GpuError::CudaError(format!("cuMemAlloc failed: {res:?}")));
            }
        }

        Ok(DevSlice::new(dptr as *mut c_void, len, cuda_free))
    }

    fn upload_slice<T: Copy + Send + Sync>(
        &self,
        host: &[T],
    ) -> Result<Self::DevSlice<T>, GpuError> {
        let bytes = host.len() * std::mem::size_of::<T>();
        let mut dptr: CUdeviceptr = 0;

        unsafe {
            let res = cuMemAlloc_v2(&mut dptr, bytes);
            if !ok(res) {
                return Err(GpuError::CudaError(format!("cuMemAlloc failed: {res:?}")));
            }

            let res = cuMemcpyHtoD_v2(dptr, host.as_ptr() as *const c_void, bytes);
            if !ok(res) {
                cuMemFree_v2(dptr);
                return Err(GpuError::CudaError(format!("cuMemcpyHtoD failed: {res:?}")));
            }
        }

        Ok(DevSlice::new(dptr as *mut c_void, host.len(), cuda_free))
    }

    fn download_slice<T: Copy + Send + Sync>(
        &self,
        dev: &Self::DevSlice<T>,
        host: &mut [T],
    ) -> Result<(), GpuError> {
        assert!(
            host.len() <= dev.len,
            "download_slice: host buffer ({} elems) exceeds device slice ({} elems) — would read out of bounds",
            host.len(),
            dev.len
        );
        let bytes = host.len() * std::mem::size_of::<T>();

        unsafe {
            let res = cuMemcpyDtoH_v2(
                host.as_mut_ptr() as *mut c_void,
                dev.ptr as CUdeviceptr,
                bytes,
            );
            if !ok(res) {
                return Err(GpuError::CudaError(format!("cuMemcpyDtoH failed: {res:?}")));
            }
        }

        Ok(())
    }

    fn get_kernel(&self, name: &str) -> Result<Self::KernelHandle, String> {
        unsafe {
            let cname = CString::new(name).map_err(|e| format!("invalid kernel name: {e}"))?;
            let mut func: CUfunction = ptr::null_mut();
            let res = cuModuleGetFunction(&mut func, self.module, cname.as_ptr());
            if !ok(res) {
                return Err(format!("cuModuleGetFunction failed for {name:?}: {res:?}"));
            }
            Ok(CudaKernelHandle(func))
        }
    }

    unsafe fn launch(
        &self,
        kernel: &Self::KernelHandle,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[*mut c_void],
    ) -> Result<(), GpuError> {
        let stream = match self.stream {
            GpuStream::Cuda(s) => s,
            _ => {
                return Err(GpuError::CudaError(
                    "CUDA backend used with non-CUDA stream".into(),
                ));
            }
        };

        unsafe {
            let res = cuLaunchKernel(
                kernel.0,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0, // sharedMemBytes
                stream,
                args.as_ptr() as *mut *mut c_void,
                ptr::null_mut(), // extra
            );
            if !ok(res) {
                return Err(GpuError::CudaError(format!(
                    "cuLaunchKernel failed: {res:?}"
                )));
            }
        }

        Ok(())
    }

    fn stream(&self) -> &GpuStream {
        &self.stream
    }
}
