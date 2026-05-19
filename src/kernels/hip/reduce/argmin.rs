// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP segmented argmin FFI wrapper.
//! This calls the C++ function:
//!     awkward_hip_segmented_argmin(data, offsets, out, n_segments, dtype_code, stream)

use std::os::raw::{c_int, c_longlong, c_void};

/// Matches the dtype codes in argmin.hip.cpp
#[derive(Debug, Clone, Copy)]
pub enum HipDtype {
    F32 = 0,
    F64 = 1,
    I32 = 2,
    I64 = 3,
}

extern "C" {
    fn awkward_hip_segmented_argmin(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_longlong,
        n_segments: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void, // hipStream_t
    );
}

/// Safe Rust wrapper
pub fn hip_segmented_argmin<T>(
    data: *const T,
    offsets: *const i64,
    out: *mut i64,
    n_segments: i64,
    dtype: HipDtype,
    stream: *mut c_void, // hipStream_t
) {
    unsafe {
        awkward_hip_segmented_argmin(
            data as *const c_void,
            offsets as *const c_longlong,
            out as *mut c_longlong,
            n_segments as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
