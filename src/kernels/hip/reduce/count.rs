// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP segmented count dispatch.
//!
//! out[seg] = offsets[seg+1] - offsets[seg].
//! Single kernel (output always i64 regardless of data type; no data pointer needed).

use crate::backend::GpuBackend;
use crate::backend::device_slice::DevicePtr;
use crate::backend::error::GpuError;
use crate::hip_args;

const BLOCK: u32 = 256;

#[inline]
fn blocks(n_segments: i64) -> u32 {
    ((n_segments as u64 + BLOCK as u64 - 1) / BLOCK as u64).max(1) as u32
}

/// Count elements per segment.
///
/// Takes only `offsets` and `out` — no data pointer, since the result is
/// purely a function of the segment boundaries.
pub fn segmented_count<B: GpuBackend>(
    backend: &B,
    offsets: &B::DevSlice<i64>,
    out: &mut B::DevSlice<i64>,
    n_segments: i64,
) -> Result<(), GpuError> {
    if n_segments == 0 {
        return Ok(());
    }
    let kernel = backend
        .get_kernel("segmented_count")
        .map_err(GpuError::HipError)?;
    hip_args!(args; offsets.as_device_ptr(), out.as_device_ptr(), n_segments);
    unsafe { backend.launch(&kernel, (blocks(n_segments), 1, 1), (BLOCK, 1, 1), &args) }
    Ok(())
}
