// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array getitem-next-array FFI wrappers (two functions).
//!
//! `*out_error`: 1=stop<start, 2=stop>lencontent, 3=index OOB.

use std::os::raw::{c_int, c_longlong, c_void};

#[repr(i32)]
pub enum NextArrayDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_list_array_getitem_next_array(
        tocarry: *mut c_longlong,
        toadvanced: *mut c_longlong,
        fromstarts: *const c_void,
        fromstops: *const c_void,
        fromarray: *const c_longlong,
        lenarray: c_longlong,
        lenstarts: c_longlong,
        lencontent: c_longlong,
        dtype_code: c_int,
        out_error: *mut c_int,
        stream: *mut c_void,
    );
    fn awkward_hip_list_array_getitem_next_array_advanced(
        tocarry: *mut c_longlong,
        toadvanced: *mut c_longlong,
        fromstarts: *const c_void,
        fromstops: *const c_void,
        fromarray: *const c_longlong,
        fromadvanced: *const c_longlong,
        lenstarts: c_longlong,
        lencontent: c_longlong,
        dtype_code: c_int,
        out_error: *mut c_int,
        stream: *mut c_void,
    );
}

pub fn hip_list_array_getitem_next_array(
    tocarry: *mut i64,
    toadvanced: *mut i64,
    fromstarts: *const c_void,
    fromstops: *const c_void,
    fromarray: *const i64,
    lenarray: i64,
    lenstarts: i64,
    lencontent: i64,
    dtype: NextArrayDtype,
    out_error: *mut i32,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_getitem_next_array(
            tocarry as *mut c_longlong,
            toadvanced as *mut c_longlong,
            fromstarts,
            fromstops,
            fromarray as *const c_longlong,
            lenarray as c_longlong,
            lenstarts as c_longlong,
            lencontent as c_longlong,
            dtype as c_int,
            out_error as *mut c_int,
            stream,
        );
    }
}

pub fn hip_list_array_getitem_next_array_advanced(
    tocarry: *mut i64,
    toadvanced: *mut i64,
    fromstarts: *const c_void,
    fromstops: *const c_void,
    fromarray: *const i64,
    fromadvanced: *const i64,
    lenstarts: i64,
    lencontent: i64,
    dtype: NextArrayDtype,
    out_error: *mut i32,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_getitem_next_array_advanced(
            tocarry as *mut c_longlong,
            toadvanced as *mut c_longlong,
            fromstarts,
            fromstops,
            fromarray as *const c_longlong,
            fromadvanced as *const c_longlong,
            lenstarts as c_longlong,
            lencontent as c_longlong,
            dtype as c_int,
            out_error as *mut c_int,
            stream,
        );
    }
}
