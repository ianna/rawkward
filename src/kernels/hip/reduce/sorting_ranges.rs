// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP segmented sorting-ranges FFI wrapper.
//! Calls: awkward_hip_segmented_sorting_ranges(data, offsets, out, n_segments, dtype_code, stream)
//!
//! **Precondition**: values within each segment must be sorted in ascending
//! order (e.g. after `hip_segmented_sort`).
//!
//! For every position `i` in the flat data array, `out[i]` receives the
//! global index of the first element of the run of equal values that
//! contains position `i`.
//!
//! Example (segment starting at global offset 5): `[1, 1, 2, 3, 3]`
//! → `out = [5, 5, 7, 8, 8]`
//!
//! `out` must have the same number of elements as `data`.

use crate::kernels::hip::reduce::HipDtype;
use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_segmented_sorting_ranges(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_longlong,
        n_segments: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

/// For each position in sorted segment data, record the global start of its
/// equal-value run.
///
/// `out` must have `offsets[n_segments]` elements (same length as `data`).
pub fn hip_segmented_sorting_ranges<T>(
    data: *const T,
    offsets: *const i64,
    out: *mut i64,
    n_segments: i64,
    dtype: HipDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_segmented_sorting_ranges(
            data as *const c_void,
            offsets as *const c_longlong,
            out as *mut c_longlong,
            n_segments as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
