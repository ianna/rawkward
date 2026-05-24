// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP segmented argmax dispatch.
//!
//! Returns the absolute index of the maximum element in each segment.
//! Empty segments write -1.  Dispatched via the `GpuBackend` abstraction,
//! identical in style to `sum.rs` and `argsort.rs`.

use crate::backend::GpuBackend;
use crate::backend::device_slice::DevicePtr;
use crate::backend::error::GpuError;
use crate::hip_args;

const BLOCK: u32 = 256;

#[inline]
fn blocks(n_segments: i64) -> u32 {
    ((n_segments as u64 + BLOCK as u64 - 1) / BLOCK as u64).max(1) as u32
}

pub fn segmented_argmax_f32<B: GpuBackend>(
    backend: &B,
    data: &B::DevSlice<f32>,
    offsets: &B::DevSlice<i64>,
    out: &mut B::DevSlice<i64>,
    n_segments: i64,
) -> Result<(), GpuError> {
    if n_segments == 0 {
        return Ok(());
    }
    let kernel = backend
        .get_kernel("segmented_argmax_f32")
        .map_err(GpuError::HipError)?;
    hip_args!(args; data.as_device_ptr(), offsets.as_device_ptr(), out.as_device_ptr(), n_segments);
    unsafe { backend.launch(&kernel, (blocks(n_segments), 1, 1), (BLOCK, 1, 1), &args) }
    Ok(())
}

pub fn segmented_argmax_f64<B: GpuBackend>(
    backend: &B,
    data: &B::DevSlice<f64>,
    offsets: &B::DevSlice<i64>,
    out: &mut B::DevSlice<i64>,
    n_segments: i64,
) -> Result<(), GpuError> {
    if n_segments == 0 {
        return Ok(());
    }
    let kernel = backend
        .get_kernel("segmented_argmax_f64")
        .map_err(GpuError::HipError)?;
    hip_args!(args; data.as_device_ptr(), offsets.as_device_ptr(), out.as_device_ptr(), n_segments);
    unsafe { backend.launch(&kernel, (blocks(n_segments), 1, 1), (BLOCK, 1, 1), &args) }
    Ok(())
}

pub fn segmented_argmax_i32<B: GpuBackend>(
    backend: &B,
    data: &B::DevSlice<i32>,
    offsets: &B::DevSlice<i64>,
    out: &mut B::DevSlice<i64>,
    n_segments: i64,
) -> Result<(), GpuError> {
    if n_segments == 0 {
        return Ok(());
    }
    let kernel = backend
        .get_kernel("segmented_argmax_i32")
        .map_err(GpuError::HipError)?;
    hip_args!(args; data.as_device_ptr(), offsets.as_device_ptr(), out.as_device_ptr(), n_segments);
    unsafe { backend.launch(&kernel, (blocks(n_segments), 1, 1), (BLOCK, 1, 1), &args) }
    Ok(())
}

pub fn segmented_argmax_i64<B: GpuBackend>(
    backend: &B,
    data: &B::DevSlice<i64>,
    offsets: &B::DevSlice<i64>,
    out: &mut B::DevSlice<i64>,
    n_segments: i64,
) -> Result<(), GpuError> {
    if n_segments == 0 {
        return Ok(());
    }
    let kernel = backend
        .get_kernel("segmented_argmax_i64")
        .map_err(GpuError::HipError)?;
    hip_args!(args; data.as_device_ptr(), offsets.as_device_ptr(), out.as_device_ptr(), n_segments);
    unsafe { backend.launch(&kernel, (blocks(n_segments), 1, 1), (BLOCK, 1, 1), &args) }
    Ok(())
}
