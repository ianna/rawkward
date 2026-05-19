// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array rpad-axis1 FFI wrapper.
//!
//! Right-pads each list to `target`, writing `toindex` (IndexedOptionArray
//! over original content) and new `tostarts`/`tostops`.
//!
//! `dtype` applies to `fromstarts`/`fromstops`/`tostarts`/`tostops` (i32/u32/i64).

use std::os::raw::{c_int, c_longlong, c_void};

#[repr(i32)]
pub enum RpadAxis1Dtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_list_array_rpad_axis1(
        toindex: *mut c_longlong,
        fromstarts: *const c_void,
        fromstops: *const c_void,
        tostarts: *mut c_void,
        tostops: *mut c_void,
        target: c_longlong,
        length: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

pub fn hip_list_array_rpad_axis1(
    toindex: *mut i64,
    fromstarts: *const c_void,
    fromstops: *const c_void,
    tostarts: *mut c_void,
    tostops: *mut c_void,
    target: i64,
    length: i64,
    dtype: RpadAxis1Dtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_rpad_axis1(
            toindex as *mut c_longlong,
            fromstarts,
            fromstops,
            tostarts,
            tostops,
            target as c_longlong,
            length as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
