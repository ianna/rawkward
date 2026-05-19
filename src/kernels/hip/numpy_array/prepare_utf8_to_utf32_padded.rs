// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP prepare-utf8-to-utf32-padded FFI wrapper.
//! Calls: awkward_hip_prepare_utf8_to_utf32_padded(fromptr, fromoffsets,
//!                                                  offsetslength, out, stream)
//!
//! Scans all UTF-8 sublists and writes the maximum codepoint count across any
//! single sublist to `*out` (device pointer).
//!
//! **The caller must zero-initialise `*out` before this call.**
//! The kernel uses `atomicMax` to accumulate the result.
//!
//! This value is the `maxcodepoints` argument needed by
//! `hip_utf8_to_utf32_padded`.

use std::os::raw::{c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_prepare_utf8_to_utf32_padded(
        fromptr:       *const u8,
        fromoffsets:   *const c_longlong,
        offsetslength: c_longlong,
        out:           *mut c_longlong,   // device pointer; zero-init before call
        stream:        *mut c_void,
    );
}

/// Find the maximum codepoint count across all UTF-8 sublists.
///
/// `out` must point to a device-allocated `i64` initialised to `0`.
/// After the call, `*out` holds the maximum.
pub fn hip_prepare_utf8_to_utf32_padded(
    fromptr:       *const u8,
    fromoffsets:   *const i64,
    offsetslength: i64,
    out:           *mut i64,
    stream:        *mut c_void,
) {
    unsafe {
        awkward_hip_prepare_utf8_to_utf32_padded(
            fromptr,
            fromoffsets as *const c_longlong,
            offsetslength as c_longlong,
            out as *mut c_longlong,
            stream,
        );
    }
}
