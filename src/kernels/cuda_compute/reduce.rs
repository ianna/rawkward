// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! CCCL/CUB segmented-reduce dispatch (option B).
//!
//! These are the Rust counterparts of `src/kernels/cuda/reduce/*` but instead of
//! launching hand-written per-segment kernels they call the host-orchestrated
//! CUB `DeviceSegmentedReduce` wrappers in `reduce.cu` (`extern "C"`). This
//! mirrors awkward-array, whose CUDA reducers route through `cuda.compute`
//! (CCCL) rather than bespoke kernels.
//!
//! Each function is generic over `B: GpuBackend` only for the device-pointer and
//! stream plumbing; the actual reduction runs entirely inside CUB on the device.
//! The stream is extracted from `backend.stream()` and passed to CUB as the same
//! handle the backend created (`CUstream` and `cudaStream_t` are the same
//! underlying type).
//!
//! The whole module is gated on `cfg(cuda_toolkit)` (set by build.rs only when
//! nvcc + the driver were found and `reduce.cu` was compiled and linked), so the
//! `GpuStream::Cuda` variant and the extern symbols always exist here.

use std::ffi::c_void;

use crate::backend::GpuBackend;
use crate::backend::GpuStream;
use crate::backend::device_slice::DevicePtr;
use crate::backend::error::GpuError;

// C ABI exported by `src/kernels/cuda_compute/reduce.cu`, compiled by build.rs
// into a static archive and linked with `-lcudart`. Every function returns the
// CUDA error code (0 == success).
unsafe extern "C" {
    fn rawkward_cccl_sum_f32(
        d_in: *const f32,
        d_off: *const i64,
        d_out: *mut f32,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_sum_f64(
        d_in: *const f64,
        d_off: *const i64,
        d_out: *mut f64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_sum_i32(
        d_in: *const i32,
        d_off: *const i64,
        d_out: *mut i32,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_sum_i64(
        d_in: *const i64,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;

    fn rawkward_cccl_min_f32(
        d_in: *const f32,
        d_off: *const i64,
        d_out: *mut f32,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_min_f64(
        d_in: *const f64,
        d_off: *const i64,
        d_out: *mut f64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_min_i32(
        d_in: *const i32,
        d_off: *const i64,
        d_out: *mut i32,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_min_i64(
        d_in: *const i64,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;

    fn rawkward_cccl_max_f32(
        d_in: *const f32,
        d_off: *const i64,
        d_out: *mut f32,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_max_f64(
        d_in: *const f64,
        d_off: *const i64,
        d_out: *mut f64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_max_i32(
        d_in: *const i32,
        d_off: *const i64,
        d_out: *mut i32,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_max_i64(
        d_in: *const i64,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;

    fn rawkward_cccl_prod_f32(
        d_in: *const f32,
        d_off: *const i64,
        d_out: *mut f32,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_prod_f64(
        d_in: *const f64,
        d_off: *const i64,
        d_out: *mut f64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_prod_i32(
        d_in: *const i32,
        d_off: *const i64,
        d_out: *mut i32,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_prod_i64(
        d_in: *const i64,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;

    fn rawkward_cccl_countnonzero_f32(
        d_in: *const f32,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_countnonzero_f64(
        d_in: *const f64,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_countnonzero_i32(
        d_in: *const i32,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_countnonzero_i64(
        d_in: *const i64,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;

    fn rawkward_cccl_count(d_off: *const i64, d_out: *mut i64, n: i64, s: *mut c_void) -> i32;

    fn rawkward_cccl_argmin_f32(
        d_in: *const f32,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_argmin_f64(
        d_in: *const f64,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_argmin_i32(
        d_in: *const i32,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_argmin_i64(
        d_in: *const i64,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;

    fn rawkward_cccl_argmax_f32(
        d_in: *const f32,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_argmax_f64(
        d_in: *const f64,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_argmax_i32(
        d_in: *const i32,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
    fn rawkward_cccl_argmax_i64(
        d_in: *const i64,
        d_off: *const i64,
        d_out: *mut i64,
        n: i64,
        s: *mut c_void,
    ) -> i32;
}

/// Extract the backend's CUDA stream as the raw handle CUB expects.
#[inline]
fn cuda_stream<B: GpuBackend>(backend: &B) -> Result<*mut c_void, GpuError> {
    match backend.stream() {
        GpuStream::Cuda(s) => Ok(*s as *mut c_void),
        _ => Err(GpuError::CudaError(
            "cuda_compute reducers require a CUDA backend/stream".into(),
        )),
    }
}

/// Map a CUB/CUDA return code to a `Result`.
#[inline]
fn check(code: i32, what: &str) -> Result<(), GpuError> {
    if code == 0 {
        Ok(())
    } else {
        Err(GpuError::CudaError(format!(
            "{what}: CUDA/CUB error code {code}"
        )))
    }
}

/// Generates a `segmented_<op>_<ty>` dispatch fn for the numeric-output reducers
/// (sum/min/max/prod: out dtype == in dtype).
macro_rules! reduce_same_out {
    ($name:ident, $ty:ty, $ffi:ident, $label:literal) => {
        pub fn $name<B: GpuBackend>(
            backend: &B,
            data: &B::DevSlice<$ty>,
            offsets: &B::DevSlice<i64>,
            out: &mut B::DevSlice<$ty>,
            n_segments: i64,
        ) -> Result<(), GpuError> {
            if n_segments == 0 {
                return Ok(());
            }
            let s = cuda_stream(backend)?;
            let code = unsafe {
                $ffi(
                    data.as_device_ptr() as *const $ty,
                    offsets.as_device_ptr() as *const i64,
                    out.as_device_ptr() as *mut $ty,
                    n_segments,
                    s,
                )
            };
            check(code, $label)
        }
    };
}

/// Generates a `segmented_<op>_<ty>` dispatch fn for the index/count reducers
/// (countnonzero/argmin/argmax: input dtype `$ty`, output always i64).
macro_rules! reduce_i64_out {
    ($name:ident, $ty:ty, $ffi:ident, $label:literal) => {
        pub fn $name<B: GpuBackend>(
            backend: &B,
            data: &B::DevSlice<$ty>,
            offsets: &B::DevSlice<i64>,
            out: &mut B::DevSlice<i64>,
            n_segments: i64,
        ) -> Result<(), GpuError> {
            if n_segments == 0 {
                return Ok(());
            }
            let s = cuda_stream(backend)?;
            let code = unsafe {
                $ffi(
                    data.as_device_ptr() as *const $ty,
                    offsets.as_device_ptr() as *const i64,
                    out.as_device_ptr() as *mut i64,
                    n_segments,
                    s,
                )
            };
            check(code, $label)
        }
    };
}

// sum
reduce_same_out!(
    segmented_sum_f32,
    f32,
    rawkward_cccl_sum_f32,
    "cccl segmented_sum_f32"
);
reduce_same_out!(
    segmented_sum_f64,
    f64,
    rawkward_cccl_sum_f64,
    "cccl segmented_sum_f64"
);
reduce_same_out!(
    segmented_sum_i32,
    i32,
    rawkward_cccl_sum_i32,
    "cccl segmented_sum_i32"
);
reduce_same_out!(
    segmented_sum_i64,
    i64,
    rawkward_cccl_sum_i64,
    "cccl segmented_sum_i64"
);

// min
reduce_same_out!(
    segmented_min_f32,
    f32,
    rawkward_cccl_min_f32,
    "cccl segmented_min_f32"
);
reduce_same_out!(
    segmented_min_f64,
    f64,
    rawkward_cccl_min_f64,
    "cccl segmented_min_f64"
);
reduce_same_out!(
    segmented_min_i32,
    i32,
    rawkward_cccl_min_i32,
    "cccl segmented_min_i32"
);
reduce_same_out!(
    segmented_min_i64,
    i64,
    rawkward_cccl_min_i64,
    "cccl segmented_min_i64"
);

// max
reduce_same_out!(
    segmented_max_f32,
    f32,
    rawkward_cccl_max_f32,
    "cccl segmented_max_f32"
);
reduce_same_out!(
    segmented_max_f64,
    f64,
    rawkward_cccl_max_f64,
    "cccl segmented_max_f64"
);
reduce_same_out!(
    segmented_max_i32,
    i32,
    rawkward_cccl_max_i32,
    "cccl segmented_max_i32"
);
reduce_same_out!(
    segmented_max_i64,
    i64,
    rawkward_cccl_max_i64,
    "cccl segmented_max_i64"
);

// prod
reduce_same_out!(
    segmented_prod_f32,
    f32,
    rawkward_cccl_prod_f32,
    "cccl segmented_prod_f32"
);
reduce_same_out!(
    segmented_prod_f64,
    f64,
    rawkward_cccl_prod_f64,
    "cccl segmented_prod_f64"
);
reduce_same_out!(
    segmented_prod_i32,
    i32,
    rawkward_cccl_prod_i32,
    "cccl segmented_prod_i32"
);
reduce_same_out!(
    segmented_prod_i64,
    i64,
    rawkward_cccl_prod_i64,
    "cccl segmented_prod_i64"
);

// countnonzero (i64 out)
reduce_i64_out!(
    segmented_countnonzero_f32,
    f32,
    rawkward_cccl_countnonzero_f32,
    "cccl segmented_countnonzero_f32"
);
reduce_i64_out!(
    segmented_countnonzero_f64,
    f64,
    rawkward_cccl_countnonzero_f64,
    "cccl segmented_countnonzero_f64"
);
reduce_i64_out!(
    segmented_countnonzero_i32,
    i32,
    rawkward_cccl_countnonzero_i32,
    "cccl segmented_countnonzero_i32"
);
reduce_i64_out!(
    segmented_countnonzero_i64,
    i64,
    rawkward_cccl_countnonzero_i64,
    "cccl segmented_countnonzero_i64"
);

// argmin (i64 out)
reduce_i64_out!(
    segmented_argmin_f32,
    f32,
    rawkward_cccl_argmin_f32,
    "cccl segmented_argmin_f32"
);
reduce_i64_out!(
    segmented_argmin_f64,
    f64,
    rawkward_cccl_argmin_f64,
    "cccl segmented_argmin_f64"
);
reduce_i64_out!(
    segmented_argmin_i32,
    i32,
    rawkward_cccl_argmin_i32,
    "cccl segmented_argmin_i32"
);
reduce_i64_out!(
    segmented_argmin_i64,
    i64,
    rawkward_cccl_argmin_i64,
    "cccl segmented_argmin_i64"
);

// argmax (i64 out)
reduce_i64_out!(
    segmented_argmax_f32,
    f32,
    rawkward_cccl_argmax_f32,
    "cccl segmented_argmax_f32"
);
reduce_i64_out!(
    segmented_argmax_f64,
    f64,
    rawkward_cccl_argmax_f64,
    "cccl segmented_argmax_f64"
);
reduce_i64_out!(
    segmented_argmax_i32,
    i32,
    rawkward_cccl_argmax_i32,
    "cccl segmented_argmax_i32"
);
reduce_i64_out!(
    segmented_argmax_i64,
    i64,
    rawkward_cccl_argmax_i64,
    "cccl segmented_argmax_i64"
);

/// Count elements per segment (offsets difference). Dtype-independent, so it
/// takes only the offsets and an i64 output.
pub fn segmented_count<B: GpuBackend>(
    backend: &B,
    offsets: &B::DevSlice<i64>,
    out: &mut B::DevSlice<i64>,
    n_segments: i64,
) -> Result<(), GpuError> {
    if n_segments == 0 {
        return Ok(());
    }
    let s = cuda_stream(backend)?;
    let code = unsafe {
        rawkward_cccl_count(
            offsets.as_device_ptr() as *const i64,
            out.as_device_ptr() as *mut i64,
            n_segments,
            s,
        )
    };
    check(code, "cccl segmented_count")
}
