// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP arg‑sort implementation for jagged lists.
//!
//! Strategy — all three kernels use the **batched** calling convention:
//! one launch covers every list in the dataset; each kernel self-filters
//! by list length and skips lists outside its tier.
//!
//!   small  (len ∈ [1,  64]):  in‑register bitonic, grid = (min(nlists,65536),1,1), block = ( 64,1,1)
//!   medium (len ∈ [65, 256]): LDS bitonic,          grid = (min(nlists,65536),1,1), block = (256,1,1)
//!   large  (len > 256):       LDS multi-elem bitonic, grid = (256,1,1),             block = (256,1,1)
//!
//! This file provides Rust wrappers that launch the corresponding
//! HIP kernels via the GpuBackend abstraction.

use crate::backend::GpuBackend;
use crate::backend::device_slice::DevicePtr;
use crate::backend::error::GpuError;
use crate::hip_args;

pub fn argsort_small<B: GpuBackend>(
    backend: &B,
    values: &B::DevSlice<f32>,
    offsets: &B::DevSlice<i64>,
    indices: &mut B::DevSlice<i64>,
    total_size: i64,
    nlists: i64,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
) -> Result<(), GpuError> {
    let kernel = backend
        .get_kernel("argsort_small_hip")
        .map_err(GpuError::HipError)?;

    hip_args!(
        args;
        values.as_device_ptr(),
        offsets.as_device_ptr(),
        indices.as_device_ptr(),
        total_size,
        nlists
    );

    unsafe { backend.launch(&kernel, grid, block, &args) }
}

pub fn argsort_medium<B: GpuBackend>(
    backend: &B,
    values: &B::DevSlice<f32>,
    offsets: &B::DevSlice<i64>,
    indices: &mut B::DevSlice<i64>,
    total_size: i64,
    nlists: i64,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
) -> Result<(), GpuError> {
    let kernel = backend
        .get_kernel("argsort_medium_hip")
        .map_err(GpuError::HipError)?;

    hip_args!(
        args;
        values.as_device_ptr(),
        offsets.as_device_ptr(),
        indices.as_device_ptr(),
        total_size,
        nlists
    );

    unsafe { backend.launch(&kernel, grid, block, &args) }
}

pub fn argsort_large<B: GpuBackend>(
    backend: &B,
    values: &B::DevSlice<f32>,
    offsets: &B::DevSlice<i64>,
    indices: &mut B::DevSlice<i64>,
    total_size: i64,
    nlists: i64,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
) -> Result<(), GpuError> {
    let kernel = backend
        .get_kernel("argsort_large_hip")
        .map_err(GpuError::HipError)?;

    hip_args!(
        args;
        values.as_device_ptr(),
        offsets.as_device_ptr(),
        indices.as_device_ptr(),
        total_size,
        nlists
    );

    unsafe { backend.launch(&kernel, grid, block, &args) }
}
