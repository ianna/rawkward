// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array broadcast-tooffsets FFI wrapper.
//!
//! Verifies list lengths match `fromoffsets` and fills `tocarry`.
//! `*out_error`: 1=stop>lencontent, 2=offsets not monotone, 3=length mismatch.

use std::os::raw::{c_int, c_longlong, c_void};

#[repr(i32)]
pub enum BroadcastDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_list_array_broadcast_tooffsets(
        tocarry:     *mut c_longlong,
        fromoffsets: *const c_longlong,
        fromstarts:  *const c_void,
        fromstops:   *const c_void,
        nlists:      c_longlong,
        lencontent:  c_longlong,
        dtype_code:  c_int,
        out_error:   *mut c_int,
        stream:      *mut c_void,
    );
}

pub fn hip_list_array_broadcast_tooffsets(
    tocarry:     *mut i64,
    fromoffsets: *const i64,
    fromstarts:  *const c_void,
    fromstops:   *const c_void,
    nlists:      i64,
    lencontent:  i64,
    dtype:       BroadcastDtype,
    out_error:   *mut i32,
    stream:      *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_broadcast_tooffsets(
            tocarry as *mut c_longlong,
            fromoffsets as *const c_longlong,
            fromstarts,
            fromstops,
            nlists as c_longlong,
            lencontent as c_longlong,
            dtype as c_int,
            out_error as *mut c_int,
            stream,
        );
    }
}
