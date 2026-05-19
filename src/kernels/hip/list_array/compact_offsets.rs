// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array compact-offsets FFI wrapper.
//!
//! Builds `tooffsets` from `fromstarts`/`fromstops`.
//! `*out_error = 1` if any `stops[i] < starts[i]`.

use std::os::raw::{c_int, c_longlong, c_void};

#[repr(i32)]
pub enum CompactOffsetsDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_list_array_compact_offsets(
        tooffsets:  *mut c_longlong,
        fromstarts: *const c_void,
        fromstops:  *const c_void,
        length:     c_longlong,
        dtype_code: c_int,
        out_error:  *mut c_int,
        stream:     *mut c_void,
    );
}

pub fn hip_list_array_compact_offsets(
    tooffsets:  *mut i64,
    fromstarts: *const c_void,
    fromstops:  *const c_void,
    length:     i64,
    dtype:      CompactOffsetsDtype,
    out_error:  *mut i32,
    stream:     *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_compact_offsets(
            tooffsets as *mut c_longlong,
            fromstarts,
            fromstops,
            length as c_longlong,
            dtype as c_int,
            out_error as *mut c_int,
            stream,
        );
    }
}
