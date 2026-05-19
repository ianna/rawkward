// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP segmented product FFI wrapper.
//! Calls: awkward_hip_segmented_prod(data, offsets, out, n_segments, dtype_code, stream)

use std::os::raw::{c_void, c_int, c_longlong};
use crate::kernels::hip::reduce::HipDtype;

unsafe extern "C" {
    fn awkward_hip_segmented_prod(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_void,
        n_segments: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

/// Multiply all elements in each segment.
///
/// `out[seg]` = product of `data[offsets[seg] .. offsets[seg+1]]`.
/// Empty segments yield `1` (the multiplicative identity).
pub fn hip_segmented_prod<T>(
    data: *const T,
    offsets: *const i64,
    out: *mut T,
    n_segments: i64,
    dtype: HipDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_segmented_prod(
            data as *const c_void,
            offsets as *const c_longlong,
            out as *mut c_void,
            n_segments as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
