// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP pad-zero-to-length FFI wrapper.
//! Calls: awkward_hip_pad_zero_to_length(fromptr, fromoffsets, target, toptr, n_lists, stream)
//!
//! For each sublist `k`, copies `fromptr[fromoffsets[k]..fromoffsets[k+1]]`
//! into `toptr[k*target..]` and zero-fills the rest up to `target` elements.
//!
//! `toptr` must have `n_lists * target` elements.

use std::os::raw::{c_void, c_longlong};

unsafe extern "C" {
    fn awkward_hip_pad_zero_to_length(
        fromptr:     *const c_void,
        fromoffsets: *const c_longlong,
        target:      c_longlong,
        toptr:       *mut c_void,
        n_lists:     c_longlong,
        stream:      *mut c_void,
    );
}

/// Pad sublists to `target` length by copying then zero-filling.
///
/// `toptr` must have `n_lists * target` bytes allocated on the device.
pub fn hip_pad_zero_to_length(
    fromptr:     *const u8,
    fromoffsets: *const i64,
    target:      i64,
    toptr:       *mut u8,
    n_lists:     i64,
    stream:      *mut c_void,
) {
    unsafe {
        awkward_hip_pad_zero_to_length(
            fromptr as *const c_void,
            fromoffsets as *const c_longlong,
            target as c_longlong,
            toptr as *mut c_void,
            n_lists as c_longlong,
            stream,
        );
    }
}
