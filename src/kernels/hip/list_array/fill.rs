// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array fill FFI wrapper.
//!
//! `awkward_hip_list_array_fill`:
//! `tostarts[tostartsoffset+i] = fromstarts[i] + base`
//! `tostops [tostopsoffset +i] = fromstops [i] + base`

use std::os::raw::{c_int, c_longlong, c_void};

#[repr(i32)]
pub enum FillDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_list_array_fill(
        tostarts: *mut c_longlong,
        tostartsoffset: c_longlong,
        tostops: *mut c_longlong,
        tostopsoffset: c_longlong,
        fromstarts: *const c_void,
        fromstops: *const c_void,
        base: c_longlong,
        length: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

pub fn hip_list_array_fill(
    tostarts: *mut i64,
    tostartsoffset: i64,
    tostops: *mut i64,
    tostopsoffset: i64,
    fromstarts: *const c_void,
    fromstops: *const c_void,
    base: i64,
    length: i64,
    dtype: FillDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_fill(
            tostarts as *mut c_longlong,
            tostartsoffset as c_longlong,
            tostops as *mut c_longlong,
            tostopsoffset as c_longlong,
            fromstarts,
            fromstops,
            base as c_longlong,
            length as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
