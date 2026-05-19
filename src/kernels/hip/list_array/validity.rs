// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array validity FFI wrapper.
//!
//! Validates starts/stops arrays of a ListArray.
//! `*out_error`: 1=start>stop, 2=start<0, 3=stop>lencontent.

use std::os::raw::{c_int, c_longlong, c_void};

#[repr(i32)]
pub enum ValidityDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_list_array_validity(
        fromstarts:  *const c_void,
        fromstops:   *const c_void,
        length:      c_longlong,
        lencontent:  c_longlong,
        dtype_code:  c_int,
        out_error:   *mut c_int,
        stream:      *mut c_void,
    );
}

pub fn hip_list_array_validity(
    fromstarts: *const c_void,
    fromstops:  *const c_void,
    length:     i64,
    lencontent: i64,
    dtype:      ValidityDtype,
    out_error:  *mut i32,
    stream:     *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_validity(
            fromstarts,
            fromstops,
            length as c_longlong,
            lencontent as c_longlong,
            dtype as c_int,
            out_error as *mut c_int,
            stream,
        );
    }
}
