// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array rpad-and-clip length (axis=1) FFI wrapper.
//!
//! Accumulates `sum(max(target, stops[i]-starts[i]))` into `*out_total`.
//! Caller must zero-init `*out_total` before launch.

use std::os::raw::{c_int, c_longlong, c_void};

#[repr(i32)]
pub enum RpadClipLenDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_list_array_rpad_and_clip_length_axis1(
        fromstarts: *const c_void,
        fromstops:  *const c_void,
        length:     c_longlong,
        target:     c_longlong,
        dtype_code: c_int,
        out_total:  *mut c_longlong,
        stream:     *mut c_void,
    );
}

pub fn hip_list_array_rpad_and_clip_length_axis1(
    fromstarts: *const c_void,
    fromstops:  *const c_void,
    length:     i64,
    target:     i64,
    dtype:      RpadClipLenDtype,
    out_total:  *mut i64,
    stream:     *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_rpad_and_clip_length_axis1(
            fromstarts,
            fromstops,
            length as c_longlong,
            target as c_longlong,
            dtype as c_int,
            out_total as *mut c_longlong,
            stream,
        );
    }
}
