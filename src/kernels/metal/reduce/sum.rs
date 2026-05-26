// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Metal GPU segmented reduce-sum.
//!
//! The MSL kernels live in `kernels.metal`.  This module provides the Rust
//! dispatch layer: it obtains a compiled pipeline state from
//! [`MetalKernelRegistry`], encodes the compute work into a command buffer,
//! and blocks until the GPU has completed.
//!
//! # Kernel semantics
//!
//! For each segment index `g` in `0..n_segments`:
//!
//! ```text
//! out[g] = Σ data[offsets[g]..offsets[g+1]]
//! ```
//!
//! Matches the CPU kernel in `src/kernels/cpu/reduce/sum.rs` and the HIP
//! kernel in `src/kernels/hip/reduce/sum.rs`.
//!
//! # Buffer layout
//!
//! | Slot | Content                    | Rust type    |
//! |------|----------------------------|--------------|
//! | 0    | Input data                 | `T`          |
//! | 1    | Segment offsets (`i64`)    | `i64`        |
//! | 2    | Output sums                | `T`          |
//!
//! All buffers must be `StorageModeShared` (the only mode `MetalBackend` uses).
//!
//! # Dispatch geometry
//!
//! Launched with `dispatch_threads({ n_segments, 1, 1 }, { tg_size, 1, 1 })`.
//! Each thread handles exactly one segment serially, so no in-shader bounds
//! check or shared-memory synchronisation is required.  The threadgroup size
//! is derived from the pipeline's `max_total_threads_per_threadgroup`, capped
//! at 256 to leave headroom for future kernels that do use shared memory.

use metal::{Buffer, MTLSize};

use crate::backend::DevSlice;
use crate::backend::metal::MetalBackend;
use crate::backend::metal::buffer_of;

// ---------------------------------------------------------------------------
// Dtype tag
// ---------------------------------------------------------------------------

/// Selects the MSL specialisation of the reduce-sum kernel.
///
/// `F64` is intentionally absent: Metal does not support `double` on any
/// Apple GPU.  Use [`segmented_sum_f64`] for a runtime error with a clear
/// message instead of a silent compile-time library failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalDtype {
    F32,
    I32,
    I64,
}

impl MetalDtype {
    /// Returns the Metal function name as it appears in `kernels.metal`.
    #[inline]
    pub fn kernel_name(self) -> &'static str {
        match self {
            MetalDtype::F32 => "reduce_sum_f32",
            MetalDtype::I32 => "reduce_sum_i32",
            MetalDtype::I64 => "reduce_sum_i64",
        }
    }
}

// ---------------------------------------------------------------------------
// Low-level dispatch (takes raw Metal Buffer refs)
// ---------------------------------------------------------------------------

/// Dispatch the segmented-sum kernel.  All three buffers must have been
/// allocated on `device` with `StorageModeShared`.
///
/// Blocks the calling thread until the GPU completes (`wait_until_completed`).
fn dispatch_segmented_sum(
    backend: &MetalBackend,
    data: &Buffer,
    offsets: &Buffer,
    out: &Buffer,
    n_segments: u64,
    dtype: MetalDtype,
) -> Result<(), String> {
    if n_segments == 0 {
        return Ok(());
    }

    let MetalBackend {
        device,
        queue,
        registry,
        ..
    } = backend;
    let pipeline = registry.get_pipeline(device, dtype.kernel_name())?;

    // Optimal threadgroup width: the pipeline's hardware maximum, capped so
    // we leave headroom for kernels that will use threadgroup shared memory.
    let tg_width = pipeline.max_total_threads_per_threadgroup().min(256);

    let threads_per_grid = MTLSize {
        width: n_segments,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: tg_width,
        height: 1,
        depth: 1,
    };

    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(data), 0);
    encoder.set_buffer(1, Some(offsets), 0);
    encoder.set_buffer(2, Some(out), 0);

    // `dispatch_threads` (non-uniform, Metal 2.0+) launches exactly
    // `threads_per_grid` threads without requiring the caller to pad to a
    // threadgroup multiple, and without needing an in-shader bounds check.
    encoder.dispatch_threads(threads_per_grid, threads_per_tg);
    encoder.end_encoding();

    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    Ok(())
}

// ---------------------------------------------------------------------------
// Public API — typed wrappers over DevSlice
// ---------------------------------------------------------------------------

/// Sum `f32` data per segment.
///
/// # Safety
///
/// `data`, `offsets`, and `out` must all have been created by `backend`
/// (i.e. they are `MetalBackend` slices with `StorageModeShared` buffers).
pub unsafe fn segmented_sum_f32(
    backend: &MetalBackend,
    data: &DevSlice<f32>,
    offsets: &DevSlice<i64>,
    out: &mut DevSlice<f32>,
    n_segments: u64,
) -> Result<(), String> {
    // Guard here, not only inside dispatch_segmented_sum: Rust evaluates all
    // function arguments before the call, so buffer_of(data/offsets/out) would
    // be invoked with null free_data on zero-length DevSlices before the inner
    // check could fire.
    if n_segments == 0 {
        return Ok(());
    }
    unsafe {
        dispatch_segmented_sum(
            backend,
            buffer_of(data),
            buffer_of(offsets),
            buffer_of(&*out),
            n_segments,
            MetalDtype::F32,
        )
    }
}

/// Sum `f64` data per segment.
///
/// **Not supported on Metal.**  Metal GPUs do not expose `double`
/// (64-bit float) in compute shaders.  This function always returns `Err`
/// so callers receive a clear message rather than a cryptic library-
/// compilation failure.
///
/// To accumulate in higher precision on Metal, consider summing in `f32`
/// with Kahan compensation, or falling back to the CPU kernel.
#[allow(clippy::too_many_arguments)]
#[allow(unused_variables)]
pub fn segmented_sum_f64(
    backend: &MetalBackend,
    data: &DevSlice<f64>,
    offsets: &DevSlice<i64>,
    out: &mut DevSlice<f64>,
    n_segments: u64,
) -> Result<(), String> {
    Err(
        "segmented_sum_f64: Metal does not support double-precision (f64) \
         compute shaders on any Apple GPU. Use the CPU kernel instead."
            .to_string(),
    )
}

/// Sum `i32` data per segment.
///
/// # Safety
///
/// Same contract as [`segmented_sum_f32`].
pub fn segmented_sum_i32(
    backend: &MetalBackend,
    data: &DevSlice<i32>,
    offsets: &DevSlice<i64>,
    out: &mut DevSlice<i32>,
    n_segments: u64,
) -> Result<(), String> {
    if n_segments == 0 {
        return Ok(());
    }
    unsafe {
        dispatch_segmented_sum(
            backend,
            buffer_of(data),
            buffer_of(offsets),
            buffer_of(&*out),
            n_segments,
            MetalDtype::I32,
        )
    }
}

/// Sum `i64` data per segment.
///
/// # Safety
///
/// Same contract as [`segmented_sum_f32`].
pub unsafe fn segmented_sum_i64(
    backend: &MetalBackend,
    data: &DevSlice<i64>,
    offsets: &DevSlice<i64>,
    out: &mut DevSlice<i64>,
    n_segments: u64,
) -> Result<(), String> {
    if n_segments == 0 {
        return Ok(());
    }
    unsafe {
        dispatch_segmented_sum(
            backend,
            buffer_of(data),
            buffer_of(offsets),
            buffer_of(&*out),
            n_segments,
            MetalDtype::I64,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::backend::{GpuBackend, metal::MetalBackend};

    fn make_backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device not available")
    }

    /// Helper: upload a slice, run `f`, download the output.
    macro_rules! round_trip {
        ($backend:expr, $data:expr, $offsets:expr, $n:expr, $fn:ident, $T:ty) => {{
            let b = &$backend;
            let data_dev = b.upload_slice::<$T>($data);
            let offsets_dev = b.upload_slice::<i64>($offsets);
            let mut out_dev = unsafe { b.alloc_slice::<$T>($n as usize) };
            unsafe {
                {
                    $fn(b, &data_dev, &offsets_dev, &mut out_dev, $n).unwrap()
                };
            }
            let mut out = vec![<$T>::default(); $n as usize];
            b.download_slice(&out_dev, &mut out);
            out
        }};
    }

    #[test]
    fn f32_two_segments() {
        let backend = make_backend();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let offsets = [0i64, 2, 4];
        let result = round_trip!(backend, &data, &offsets, 2, segmented_sum_f32, f32);
        assert!((result[0] - 3.0).abs() < 1e-6, "got {}", result[0]);
        assert!((result[1] - 7.0).abs() < 1e-6, "got {}", result[1]);
    }

    #[test]
    fn f64_returns_unsupported_error() {
        // Metal does not support double-precision shaders on any Apple GPU.
        // segmented_sum_f64 must return Err rather than crashing or silently
        // producing wrong results.
        let backend = make_backend();
        let data = [1.5f64, 2.5, 3.0];
        let offsets = [0i64, 3];
        let data_dev = backend.upload_slice::<f64>(&data);
        let offsets_dev = backend.upload_slice::<i64>(&offsets);
        let mut out_dev = unsafe { backend.alloc_slice::<f64>(1) };
        let result = { segmented_sum_f64(&backend, &data_dev, &offsets_dev, &mut out_dev, 1) };
        assert!(result.is_err(), "expected Err for unsupported f64, got Ok");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("double-precision") || msg.contains("f64"),
            "error message should mention f64/double: {msg}"
        );
    }

    #[test]
    #[allow(unused_unsafe)]
    fn i32_empty_segment() {
        let backend = make_backend();
        let data = [10i32, 20];
        let offsets = [0i64, 2, 2]; // second segment is empty
        let result = round_trip!(backend, &data, &offsets, 2, segmented_sum_i32, i32);
        assert_eq!(result[0], 30);
        assert_eq!(result[1], 0);
    }

    #[test]
    fn i64_many_segments() {
        let backend = make_backend();
        // 4 segments of 1 element each
        let data = [1i64, 2, 3, 4];
        let offsets = [0i64, 1, 2, 3, 4];
        let result = round_trip!(backend, &data, &offsets, 4, segmented_sum_i64, i64);
        assert_eq!(result, [1, 2, 3, 4]);
    }

    #[test]
    fn zero_segments_is_noop() {
        let backend = make_backend();
        let data: [f32; 0] = [];
        let offsets = [0i64];
        let result = round_trip!(backend, &data, &offsets, 0, segmented_sum_f32, f32);
        assert!(result.is_empty());
    }
}
