// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP segmented unique-ranges FFI wrapper.
//! Calls: awkward_hip_segmented_unique_ranges(data, offsets, out, n_segments, dtype_code, stream)
//!
//! **Precondition**: values within each segment must be sorted in ascending
//! order (e.g. after `hip_segmented_sort`).
//!
//! For every position `i` in the flat data array, `out[i]` receives the
//! 0-based index of the run of equal values that position `i` belongs to
//! within its segment.
//!
//! Example: sorted segment `[1, 1, 2, 3, 3]` → `out = [0, 0, 1, 2, 2]`.
//!
//! Together with `hip_segmented_unique_offsets` (which gives the number of
//! runs per segment), this is sufficient to scatter every element to its
//! unique-value slot without a second pass over the data.
//!
//! `out` must have `offsets[n_segments]` elements (same length as `data`).

use crate::kernels::hip::reduce::HipDtype;
use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_segmented_unique_ranges(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_longlong,
        n_segments: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

/// Assign a 0-based run index to every position in sorted segment data.
///
/// `out[i]` = index of the equal-value run that position `i` belongs to
/// within its segment.  `out` must have `offsets[n_segments]` elements.
pub fn hip_segmented_unique_ranges<T>(
    data: *const T,
    offsets: *const i64,
    out: *mut i64,
    n_segments: i64,
    dtype: HipDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_segmented_unique_ranges(
            data as *const c_void,
            offsets as *const c_longlong,
            out as *mut c_longlong,
            n_segments as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
