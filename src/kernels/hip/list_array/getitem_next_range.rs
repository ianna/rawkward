// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array getitem-next-range FFI wrappers (four functions).
//!
//! Pass `i64::MIN` for `start`/`stop` to indicate Python `None`.

use std::os::raw::{c_int, c_longlong, c_void};

#[repr(i32)]
pub enum RangeDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_list_array_getitem_next_range_carrylength(
        fromstarts: *const c_void,
        fromstops: *const c_void,
        length: c_longlong,
        start: c_longlong,
        stop: c_longlong,
        step: c_longlong,
        dtype_code: c_int,
        out_total: *mut c_longlong,
        stream: *mut c_void,
    );
    fn awkward_hip_list_array_getitem_next_range_counts(
        fromoffsets: *const c_void,
        lenstarts: c_longlong,
        dtype_code: c_int,
        out_total: *mut c_longlong,
        stream: *mut c_void,
    );
    fn awkward_hip_list_array_getitem_next_range(
        tooffsets: *mut c_longlong,
        tocarry: *mut c_longlong,
        fromstarts: *const c_void,
        fromstops: *const c_void,
        length: c_longlong,
        start: c_longlong,
        stop: c_longlong,
        step: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
    fn awkward_hip_list_array_getitem_next_range_spreadadvanced(
        toadvanced: *mut c_longlong,
        fromadvanced: *const c_longlong,
        fromoffsets: *const c_void,
        lenstarts: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

pub fn hip_list_array_getitem_next_range_carrylength(
    fromstarts: *const c_void,
    fromstops: *const c_void,
    length: i64,
    start: i64, // i64::MIN = None
    stop: i64,  // i64::MIN = None
    step: i64,
    dtype: RangeDtype,
    out_total: *mut i64,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_getitem_next_range_carrylength(
            fromstarts,
            fromstops,
            length as c_longlong,
            start as c_longlong,
            stop as c_longlong,
            step as c_longlong,
            dtype as c_int,
            out_total as *mut c_longlong,
            stream,
        );
    }
}

pub fn hip_list_array_getitem_next_range_counts(
    fromoffsets: *const c_void,
    lenstarts: i64,
    dtype: RangeDtype,
    out_total: *mut i64,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_getitem_next_range_counts(
            fromoffsets,
            lenstarts as c_longlong,
            dtype as c_int,
            out_total as *mut c_longlong,
            stream,
        );
    }
}

pub fn hip_list_array_getitem_next_range(
    tooffsets: *mut i64,
    tocarry: *mut i64,
    fromstarts: *const c_void,
    fromstops: *const c_void,
    length: i64,
    start: i64, // i64::MIN = None
    stop: i64,  // i64::MIN = None
    step: i64,
    dtype: RangeDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_getitem_next_range(
            tooffsets as *mut c_longlong,
            tocarry as *mut c_longlong,
            fromstarts,
            fromstops,
            length as c_longlong,
            start as c_longlong,
            stop as c_longlong,
            step as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}

pub fn hip_list_array_getitem_next_range_spreadadvanced(
    toadvanced: *mut i64,
    fromadvanced: *const i64,
    fromoffsets: *const c_void,
    lenstarts: i64,
    dtype: RangeDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_getitem_next_range_spreadadvanced(
            toadvanced as *mut c_longlong,
            fromadvanced as *const c_longlong,
            fromoffsets,
            lenstarts as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
