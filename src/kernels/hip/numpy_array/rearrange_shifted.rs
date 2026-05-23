// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP rearrange-shifted FFI wrapper.
//! Calls: awkward_hip_rearrange_shifted(toptr, fromshifts, fromoffsets,
//!                                      fromparents, fromstarts,
//!                                      offsetslength, length, stream)
//!
//! Two-pass in-place adjustment of `toptr` (all device pointers):
//!
//! **Pass 1** — for each segment `i`, add `fromoffsets[i]` to every element
//! of `toptr` in that segment's block.
//!
//! **Pass 2** — for each element `j`:
//! `toptr[j] += fromshifts[toptr[j]] - fromstarts[fromparents[j]]`
//!
//! `offsetslength = n_lists + 1`;  `length` = total elements in `toptr`.

use std::os::raw::{c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_rearrange_shifted(
        toptr: *mut c_longlong,
        fromshifts: *const c_longlong,
        fromoffsets: *const c_longlong,
        fromparents: *const c_longlong,
        fromstarts: *const c_longlong,
        offsetslength: c_longlong,
        length: c_longlong,
        stream: *mut c_void,
    );
}

/// Apply the two-pass rearrange-shifted transformation to `toptr`.
pub fn hip_rearrange_shifted(
    toptr: *mut i64,
    fromshifts: *const i64,
    fromoffsets: *const i64,
    fromparents: *const i64,
    fromstarts: *const i64,
    offsetslength: i64,
    length: i64,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_rearrange_shifted(
            toptr as *mut c_longlong,
            fromshifts as *const c_longlong,
            fromoffsets as *const c_longlong,
            fromparents as *const c_longlong,
            fromstarts as *const c_longlong,
            offsetslength as c_longlong,
            length as c_longlong,
            stream,
        );
    }
}
