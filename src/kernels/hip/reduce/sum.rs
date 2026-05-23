// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP segmented reduce-sum dispatch.
//!
//! The MSL kernels live in `sum.hip.cpp`.  This module provides the Rust
//! dispatch layer using the `GpuBackend` abstraction, identical in style to
//! `src/kernels/hip/sort/argsort.rs`.
//!
//! # Kernel semantics
//!
//! For each segment index `seg` in `0..n_segments`:
//!
//! ```text
//! out[seg] = Σ data[offsets[seg]..offsets[seg+1]]
//! ```
//!
//! # Dispatch geometry
//!
//! `blocks = ceil(n_segments / 256)`, `threads = 256`.
//! Out-of-bounds threads self-exit via an in-shader bounds check.
//!
//! # Note on f64
//!
//! Unlike Metal, AMD GCN/RDNA compute shaders fully support `double`.
//! `segmented_sum_f64` works correctly on this backend.

use crate::backend::GpuBackend;
use crate::backend::device_slice::DevicePtr;
use crate::backend::error::GpuError;
use crate::hip_args;

const BLOCK: u32 = 256;

#[inline]
fn blocks(n_segments: i64) -> u32 {
    ((n_segments as u64 + BLOCK as u64 - 1) / BLOCK as u64).max(1) as u32
}

/// Sum `f32` data per segment.
pub fn segmented_sum_f32<B: GpuBackend>(
    backend: &B,
    data: &B::DevSlice<f32>,
    offsets: &B::DevSlice<i64>,
    out: &mut B::DevSlice<f32>,
    n_segments: i64,
) -> Result<(), GpuError> {
    if n_segments == 0 {
        return Ok(());
    }
    let kernel = backend
        .get_kernel("segmented_sum_f32")
        .map_err(GpuError::HipError)?;
    hip_args!(args;
        data.as_device_ptr(),
        offsets.as_device_ptr(),
        out.as_device_ptr(),
        n_segments
    );
    unsafe { backend.launch(&kernel, (blocks(n_segments), 1, 1), (BLOCK, 1, 1), &args) }
    Ok(())
}

/// Sum `f64` data per segment.
///
/// Supported on AMD GCN and RDNA GPUs (unlike Metal, which forbids double).
pub fn segmented_sum_f64<B: GpuBackend>(
    backend: &B,
    data: &B::DevSlice<f64>,
    offsets: &B::DevSlice<i64>,
    out: &mut B::DevSlice<f64>,
    n_segments: i64,
) -> Result<(), GpuError> {
    if n_segments == 0 {
        return Ok(());
    }
    let kernel = backend
        .get_kernel("segmented_sum_f64")
        .map_err(GpuError::HipError)?;
    hip_args!(args;
        data.as_device_ptr(),
        offsets.as_device_ptr(),
        out.as_device_ptr(),
        n_segments
    );
    unsafe { backend.launch(&kernel, (blocks(n_segments), 1, 1), (BLOCK, 1, 1), &args) }
    Ok(())
}

/// Sum `i32` data per segment.
pub fn segmented_sum_i32<B: GpuBackend>(
    backend: &B,
    data: &B::DevSlice<i32>,
    offsets: &B::DevSlice<i64>,
    out: &mut B::DevSlice<i32>,
    n_segments: i64,
) -> Result<(), GpuError> {
    if n_segments == 0 {
        return Ok(());
    }
    let kernel = backend
        .get_kernel("segmented_sum_i32")
        .map_err(GpuError::HipError)?;
    hip_args!(args;
        data.as_device_ptr(),
        offsets.as_device_ptr(),
        out.as_device_ptr(),
        n_segments
    );
    unsafe { backend.launch(&kernel, (blocks(n_segments), 1, 1), (BLOCK, 1, 1), &args) }
    Ok(())
}

/// Sum `i64` data per segment.
pub fn segmented_sum_i64<B: GpuBackend>(
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
        .get_kernel("segmented_sum_i64")
        .map_err(GpuError::HipError)?;
    hip_args!(args;
        data.as_device_ptr(),
        offsets.as_device_ptr(),
        out.as_device_ptr(),
        n_segments
    );
    unsafe { backend.launch(&kernel, (blocks(n_segments), 1, 1), (BLOCK, 1, 1), &args) }
    Ok(())
}
