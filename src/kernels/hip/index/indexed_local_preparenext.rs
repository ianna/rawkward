// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP indexed-array local-preparenext FFI wrapper.
//! Calls: awkward_hip_indexed_array_local_preparenext(tocarry, parents,
//!                                                     parentslength,
//!                                                     nextparents, nextlen,
//!                                                     stream)
//!
//! For each i in 0..parentslength:
//!   if j < nextlen && parents[i] == nextparents[j]: tocarry[i] = j, j++
//!   else: tocarry[i] = -1
//!
//! Single-threaded kernel (sequential j counter with data dependency).

use std::os::raw::{c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_indexed_array_local_preparenext(
        tocarry: *mut c_longlong,
        parents: *const c_longlong,
        parentslength: c_longlong,
        nextparents: *const c_longlong,
        nextlen: c_longlong,
        stream: *mut c_void,
    );
}

/// Build carry indices by matching parents with nextparents.
///
/// Positions where no match exists receive -1.
pub fn hip_indexed_array_local_preparenext(
    tocarry: *mut i64,
    parents: *const i64,
    parentslength: i64,
    nextparents: *const i64,
    nextlen: i64,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_local_preparenext(
            tocarry as *mut c_longlong,
            parents as *const c_longlong,
            parentslength as c_longlong,
            nextparents as *const c_longlong,
            nextlen as c_longlong,
            stream,
        );
    }
}
