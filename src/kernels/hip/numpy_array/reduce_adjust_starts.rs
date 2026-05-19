// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP reduce-adjust-starts FFI wrapper.
//! Calls: awkward_hip_reduce_adjust_starts(toptr, parents, starts, length, stream)
//!
//! For each element `i` in `toptr`:
//! - If `toptr[i] >= 0`, subtract `starts[parents[toptr[i]]]` so the stored
//!   global flat index becomes a within-list local position.
//! - Sentinels (`-1`) are left unchanged.
//!
//! Called after `argmin`/`argmax` to normalise results.

use std::os::raw::{c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_reduce_adjust_starts(
        toptr: *mut c_longlong,
        parents: *const c_longlong,
        starts: *const c_longlong,
        length: c_longlong,
        stream: *mut c_void,
    );
}

/// Convert global argmin/argmax indices in `toptr` to within-list positions.
pub fn hip_reduce_adjust_starts(
    toptr: *mut i64,
    parents: *const i64,
    starts: *const i64,
    length: i64,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_reduce_adjust_starts(
            toptr as *mut c_longlong,
            parents as *const c_longlong,
            starts as *const c_longlong,
            length as c_longlong,
            stream,
        );
    }
}
