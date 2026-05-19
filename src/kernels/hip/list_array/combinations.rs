// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP list-array combinations FFI wrappers (two functions).
//!
//! `awkward_hip_list_array_combinations_length` — compute per-list C(size,n)
//! counts and write cumulative offsets; accumulate grand total.
//!
//! `awkward_hip_list_array_combinations` — enumerate all n-combinations per
//! list into a flat `tocarry` array laid out as
//! `tocarry[col * total_combinations + k]`.

use std::os::raw::{c_int, c_longlong, c_void};

#[repr(i32)]
pub enum CombinationsDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_list_array_combinations_length(
        tooffsets: *mut c_longlong,
        fromstarts: *const c_void,
        fromstops: *const c_void,
        length: c_longlong,
        n: c_longlong,
        replacement: c_int,
        dtype_code: c_int,
        out_total: *mut c_longlong,
        stream: *mut c_void,
    );
    fn awkward_hip_list_array_combinations(
        tocarry: *mut c_longlong,
        fromstarts: *const c_void,
        fromstops: *const c_void,
        length: c_longlong,
        n: c_longlong,
        replacement: c_int,
        total_combinations: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

/// Compute combination counts per list and build cumulative `tooffsets`.
/// Caller must zero-init `*out_total` before launch.
pub fn hip_list_array_combinations_length(
    tooffsets: *mut i64,
    fromstarts: *const c_void,
    fromstops: *const c_void,
    length: i64,
    n: i64,
    replacement: bool,
    dtype: CombinationsDtype,
    out_total: *mut i64,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_combinations_length(
            tooffsets as *mut c_longlong,
            fromstarts,
            fromstops,
            length as c_longlong,
            n as c_longlong,
            replacement as c_int,
            dtype as c_int,
            out_total as *mut c_longlong,
            stream,
        );
    }
}

/// Enumerate all n-combinations per list into `tocarry`.
/// `total_combinations` must be the grand total pre-computed by
/// [`hip_list_array_combinations_length`].
pub fn hip_list_array_combinations(
    tocarry: *mut i64,
    fromstarts: *const c_void,
    fromstops: *const c_void,
    length: i64,
    n: i64,
    replacement: bool,
    total_combinations: i64,
    dtype: CombinationsDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_list_array_combinations(
            tocarry as *mut c_longlong,
            fromstarts,
            fromstops,
            length as c_longlong,
            n as c_longlong,
            replacement as c_int,
            total_combinations as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
