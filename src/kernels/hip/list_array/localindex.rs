// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array localindex FFI wrapper.
//!
//! For each list `i` spanning `offsets[i]..offsets[i+1]`:
//! `toindex[j] = j - offsets[i]`

use std::os::raw::{c_int, c_longlong, c_void};

#[repr(i32)]
pub enum LocalindexDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_list_array_localindex(
        toindex: *mut c_longlong,
        offsets: *const c_void,
        length: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

pub fn hip_list_array_localindex(
    toindex: *mut i64,
    offsets: *const c_void,
    length: i64,
    dtype: LocalindexDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_localindex(
            toindex as *mut c_longlong,
            offsets,
            length as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
