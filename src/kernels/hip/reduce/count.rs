// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP segmented count FFI wrapper.
//! Calls: awkward_hip_segmented_count(data, offsets, out, n_segments, dtype_code, stream)
//!
//! `out[seg]` = number of elements in segment `seg`.
//! The `dtype` parameter is accepted for API uniformity but is not used — the
//! count is always written as `i64` regardless of input element type.

use std::os::raw::{c_void, c_int, c_longlong};
use crate::kernels::hip::reduce::HipDtype;

unsafe extern "C" {
    fn awkward_hip_segmented_count(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_longlong,
        n_segments: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

/// Count the number of elements in each segment.
///
/// `out[seg]` = `offsets[seg+1] - offsets[seg]`.
/// `out` must have at least `n_segments` elements.
pub fn hip_segmented_count<T>(
    data: *const T,
    offsets: *const i64,
    out: *mut i64,
    n_segments: i64,
    dtype: HipDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_segmented_count(
            data as *const c_void,
            offsets as *const c_longlong,
            out as *mut c_longlong,
            n_segments as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
