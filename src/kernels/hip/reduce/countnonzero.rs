// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP segmented count-nonzero FFI wrapper.
//! Calls: awkward_hip_segmented_countnonzero(data, offsets, out, n_segments, dtype_code, stream)

use crate::kernels::hip::reduce::HipDtype;
use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_segmented_countnonzero(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_longlong,
        n_segments: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

/// Count the number of non-zero elements in each segment.
///
/// `out[seg]` = number of elements in `data[offsets[seg] .. offsets[seg+1]]`
/// that compare `!= 0`.  `out` must have at least `n_segments` elements.
pub fn hip_segmented_countnonzero<T>(
    data: *const T,
    offsets: *const i64,
    out: *mut i64,
    n_segments: i64,
    dtype: HipDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_segmented_countnonzero(
            data as *const c_void,
            offsets as *const c_longlong,
            out as *mut c_longlong,
            n_segments as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
