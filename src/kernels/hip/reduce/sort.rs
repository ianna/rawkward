// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP segmented sort FFI wrapper.
//! Calls: awkward_hip_segmented_sort(data, offsets, out, n_segments, dtype_code, stream)
//!
//! Sorts the values within each segment using a three-tier batched bitonic
//! network (≤64 / ≤256 / ≤4096 elements).  The output buffer `out` has the
//! same length as `data`; `out[offsets[seg] .. offsets[seg+1]]` contains the
//! sorted values of segment `seg` in ascending order.
//! Lists longer than 4096 elements are silently left unchanged.

use std::os::raw::{c_void, c_int, c_longlong};
use crate::kernels::hip::reduce::HipDtype;

unsafe extern "C" {
    fn awkward_hip_segmented_sort(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_void,
        n_segments: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

/// Sort the values within each segment in ascending order.
///
/// `out` must have the same number of elements as `data` (i.e. `offsets[n_segments]`).
/// `out[offsets[seg] .. offsets[seg+1]]` holds the sorted values of segment `seg`.
pub fn hip_segmented_sort<T>(
    data: *const T,
    offsets: *const i64,
    out: *mut T,
    n_segments: i64,
    dtype: HipDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_segmented_sort(
            data as *const c_void,
            offsets as *const c_longlong,
            out as *mut c_void,
            n_segments as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
