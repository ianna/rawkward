// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array rpad-axis1 FFI wrapper.
//!
//! Right-pads each list to `target`, writing `toindex` (IndexedOptionArray
//! over original content) and new `tostarts`/`tostops`.

use std::os::raw::{c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_list_array_rpad_axis1(
        toindex:    *mut c_longlong,
        fromstarts: *const c_longlong,
        fromstops:  *const c_longlong,
        tostarts:   *mut c_longlong,
        tostops:    *mut c_longlong,
        target:     c_longlong,
        length:     c_longlong,
        stream:     *mut c_void,
    );
}

pub fn hip_list_array_rpad_axis1(
    toindex:    *mut i64,
    fromstarts: *const i64,
    fromstops:  *const i64,
    tostarts:   *mut i64,
    tostops:    *mut i64,
    target:     i64,
    length:     i64,
    stream:     *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_rpad_axis1(
            toindex    as *mut c_longlong,
            fromstarts as *const c_longlong,
            fromstops  as *const c_longlong,
            tostarts   as *mut c_longlong,
            tostops    as *mut c_longlong,
            target as c_longlong,
            length as c_longlong,
            stream,
        );
    }
}
