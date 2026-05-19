// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array getitem-next-at FFI wrapper.
//!
//! Resolves integer index `at` (with negative wrap-around) for each list.
//! `*out_error = 1` if any index is out of range.

use std::os::raw::{c_int, c_longlong, c_void};

#[repr(i32)]
pub enum GetitemNextAtDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_list_array_getitem_next_at(
        tocarry: *mut c_longlong,
        fromstarts: *const c_void,
        fromstops: *const c_void,
        at: c_longlong,
        length: c_longlong,
        dtype_code: c_int,
        out_error: *mut c_int,
        stream: *mut c_void,
    );
}

pub fn hip_list_array_getitem_next_at(
    tocarry: *mut i64,
    fromstarts: *const c_void,
    fromstops: *const c_void,
    at: i64,
    length: i64,
    dtype: GetitemNextAtDtype,
    out_error: *mut i32,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_getitem_next_at(
            tocarry as *mut c_longlong,
            fromstarts,
            fromstops,
            at as c_longlong,
            length as c_longlong,
            dtype as c_int,
            out_error as *mut c_int,
            stream,
        );
    }
}
