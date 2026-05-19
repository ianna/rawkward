// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP utf8-to-utf32-padded FFI wrapper.
//! Calls: awkward_hip_utf8_to_utf32_padded(fromptr, fromoffsets, offsetslength,
//!                                          maxcodepoints, toptr, out_error, stream)
//!
//! Decodes every UTF-8 sublist into padded UTF-32 codepoints.
//! For sublist `k`, writes `maxcodepoints` values to
//! `toptr[k*maxcodepoints .. (k+1)*maxcodepoints]`, zero-padding unused slots.
//!
//! `toptr` must have `(offsetslength - 1) * maxcodepoints` u32 elements.
//!
//! **Error handling**: `out_error` is a device pointer to an `i32` that the
//! caller must zero-initialise.  On invalid UTF-8, the kernel sets
//! `*out_error` to the 1-based sublist index; on success it remains `0`.
//!
//! Obtain `maxcodepoints` from `hip_prepare_utf8_to_utf32_padded`.

use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_utf8_to_utf32_padded(
        fromptr:       *const u8,
        fromoffsets:   *const c_longlong,
        offsetslength: c_longlong,
        maxcodepoints: c_longlong,
        toptr:         *mut u32,
        out_error:     *mut c_int,    // device pointer; zero-init before call
        stream:        *mut c_void,
    );
}

/// Decode UTF-8 sublists into padded UTF-32 output.
///
/// `out_error` is a device pointer to an `i32`; zero-init before the call.
/// Non-zero after the call means sublist `*out_error - 1` had invalid UTF-8.
pub fn hip_utf8_to_utf32_padded(
    fromptr:       *const u8,
    fromoffsets:   *const i64,
    offsetslength: i64,
    maxcodepoints: i64,
    toptr:         *mut u32,
    out_error:     *mut i32,
    stream:        *mut c_void,
) {
    unsafe {
        awkward_hip_utf8_to_utf32_padded(
            fromptr,
            fromoffsets as *const c_longlong,
            offsetslength as c_longlong,
            maxcodepoints as c_longlong,
            toptr,
            out_error as *mut c_int,
            stream,
        );
    }
}
