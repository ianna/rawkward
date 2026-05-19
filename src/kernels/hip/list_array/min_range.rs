// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array min-range FFI wrapper.
//!
//! Writes `min(stops[i] - starts[i])` into `*out_min` (device pointer).
//! Caller must initialise `*out_min` to `i64::MAX` before launch.

use std::os::raw::{c_int, c_longlong, c_void};

#[repr(i32)]
pub enum MinRangeDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_list_array_min_range(
        fromstarts: *const c_void,
        fromstops:  *const c_void,
        length:     c_longlong,
        dtype_code: c_int,
        out_min:    *mut c_longlong,
        stream:     *mut c_void,
    );
}

pub fn hip_list_array_min_range(
    fromstarts: *const c_void,
    fromstops:  *const c_void,
    length:     i64,
    dtype:      MinRangeDtype,
    out_min:    *mut i64,
    stream:     *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_min_range(
            fromstarts,
            fromstops,
            length as c_longlong,
            dtype as c_int,
            out_min as *mut c_longlong,
            stream,
        );
    }
}
