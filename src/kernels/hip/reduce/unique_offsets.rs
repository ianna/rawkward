// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP segmented unique-offsets FFI wrapper.
//! Calls: awkward_hip_segmented_unique_offsets(data, offsets, out, n_segments, dtype_code, stream)
//!
//! **Precondition**: values within each segment must be sorted in ascending
//! order (e.g. after `hip_segmented_sort`).
//!
//! `out[seg]` = number of distinct values in `data[offsets[seg] .. offsets[seg+1]]`.
//!
//! The caller can perform an exclusive prefix-sum on `out` to obtain a proper
//! offsets array that indexes into a compact unique-values buffer.
//!
//! Example: segment `[1, 1, 2, 3, 3]` → `out[seg] = 3`.
//!
//! `out` must have at least `n_segments` elements.

use crate::kernels::hip::reduce::HipDtype;
use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_segmented_unique_offsets(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_longlong,
        n_segments: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

/// Count the number of distinct values in each segment of sorted data.
///
/// `out[seg]` = number of unique values in `data[offsets[seg] .. offsets[seg+1]]`.
/// `out` must have at least `n_segments` elements.
pub fn hip_segmented_unique_offsets<T>(
    data: *const T,
    offsets: *const i64,
    out: *mut i64,
    n_segments: i64,
    dtype: HipDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_segmented_unique_offsets(
            data as *const c_void,
            offsets as *const c_longlong,
            out as *mut c_longlong,
            n_segments as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
