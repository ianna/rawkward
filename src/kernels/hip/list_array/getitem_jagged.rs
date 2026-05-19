// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array getitem-jagged FFI wrappers (five functions).

use std::os::raw::{c_int, c_longlong, c_void};

#[repr(i32)]
pub enum JaggedDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_list_array_getitem_jagged_apply(
        tooffsets: *mut c_longlong,
        tocarry: *mut c_longlong,
        slicestarts: *const c_longlong,
        slicestops: *const c_longlong,
        sliceindex: *const c_longlong,
        sliceinnerlen: c_longlong,
        fromstarts: *const c_void,
        fromstops: *const c_void,
        contentlen: c_longlong,
        sliceouterlen: c_longlong,
        dtype_code: c_int,
        out_error: *mut c_int,
        stream: *mut c_void,
    );
    fn awkward_hip_list_array_getitem_jagged_numvalid(
        slicestarts: *const c_longlong,
        slicestops: *const c_longlong,
        missing: *const c_longlong,
        missinglength: c_longlong,
        length: c_longlong,
        out_count: *mut c_longlong,
        out_error: *mut c_int,
        stream: *mut c_void,
    );
    fn awkward_hip_list_array_getitem_jagged_carrylen(
        slicestarts: *const c_longlong,
        slicestops: *const c_longlong,
        length: c_longlong,
        out_count: *mut c_longlong,
        stream: *mut c_void,
    );
    fn awkward_hip_list_array_getitem_jagged_shrink(
        tocarry: *mut c_longlong,
        tosmalloffsets: *mut c_longlong,
        tolargeoffsets: *mut c_longlong,
        slicestarts: *const c_longlong,
        slicestops: *const c_longlong,
        missing: *const c_longlong,
        length: c_longlong,
        out_count: *mut c_longlong,
        stream: *mut c_void,
    );
    fn awkward_hip_list_array_getitem_jagged_descend(
        tooffsets: *mut c_longlong,
        slicestarts: *const c_longlong,
        slicestops: *const c_longlong,
        fromstarts: *const c_void,
        fromstops: *const c_void,
        sliceouterlen: c_longlong,
        dtype_code: c_int,
        out_error: *mut c_int,
        stream: *mut c_void,
    );
}

pub fn hip_list_array_getitem_jagged_apply(
    tooffsets: *mut i64,
    tocarry: *mut i64,
    slicestarts: *const i64,
    slicestops: *const i64,
    sliceindex: *const i64,
    sliceinnerlen: i64,
    fromstarts: *const c_void,
    fromstops: *const c_void,
    contentlen: i64,
    sliceouterlen: i64,
    dtype: JaggedDtype,
    out_error: *mut i32,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_getitem_jagged_apply(
            tooffsets as *mut c_longlong,
            tocarry as *mut c_longlong,
            slicestarts as *const c_longlong,
            slicestops as *const c_longlong,
            sliceindex as *const c_longlong,
            sliceinnerlen as c_longlong,
            fromstarts,
            fromstops,
            contentlen as c_longlong,
            sliceouterlen as c_longlong,
            dtype as c_int,
            out_error as *mut c_int,
            stream,
        );
    }
}

pub fn hip_list_array_getitem_jagged_numvalid(
    slicestarts: *const i64,
    slicestops: *const i64,
    missing: *const i64,
    missinglength: i64,
    length: i64,
    out_count: *mut i64,
    out_error: *mut i32,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_getitem_jagged_numvalid(
            slicestarts as *const c_longlong,
            slicestops as *const c_longlong,
            missing as *const c_longlong,
            missinglength as c_longlong,
            length as c_longlong,
            out_count as *mut c_longlong,
            out_error as *mut c_int,
            stream,
        );
    }
}

pub fn hip_list_array_getitem_jagged_carrylen(
    slicestarts: *const i64,
    slicestops: *const i64,
    length: i64,
    out_count: *mut i64,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_getitem_jagged_carrylen(
            slicestarts as *const c_longlong,
            slicestops as *const c_longlong,
            length as c_longlong,
            out_count as *mut c_longlong,
            stream,
        );
    }
}

pub fn hip_list_array_getitem_jagged_shrink(
    tocarry: *mut i64,
    tosmalloffsets: *mut i64,
    tolargeoffsets: *mut i64,
    slicestarts: *const i64,
    slicestops: *const i64,
    missing: *const i64,
    length: i64,
    out_count: *mut i64,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_getitem_jagged_shrink(
            tocarry as *mut c_longlong,
            tosmalloffsets as *mut c_longlong,
            tolargeoffsets as *mut c_longlong,
            slicestarts as *const c_longlong,
            slicestops as *const c_longlong,
            missing as *const c_longlong,
            length as c_longlong,
            out_count as *mut c_longlong,
            stream,
        );
    }
}

pub fn hip_list_array_getitem_jagged_descend(
    tooffsets: *mut i64,
    slicestarts: *const i64,
    slicestops: *const i64,
    fromstarts: *const c_void,
    fromstops: *const c_void,
    sliceouterlen: i64,
    dtype: JaggedDtype,
    out_error: *mut i32,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_getitem_jagged_descend(
            tooffsets as *mut c_longlong,
            slicestarts as *const c_longlong,
            slicestops as *const c_longlong,
            fromstarts,
            fromstops,
            sliceouterlen as c_longlong,
            dtype as c_int,
            out_error as *mut c_int,
            stream,
        );
    }
}
