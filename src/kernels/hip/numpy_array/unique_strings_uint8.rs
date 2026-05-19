// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP unique-strings-uint8 FFI wrapper.
//! Calls: awkward_hip_unique_strings_uint8(toptr, offsets, offsetslength,
//!                                         outoffsets, out_count, stream)
//!
//! Deduplicates consecutive equal strings in the flat byte buffer `toptr`.
//! Kept strings are compacted in-place; `outoffsets` receives their new
//! boundary positions; `*out_count` receives the number of `outoffsets`
//! entries written (= unique string count + 1).
//!
//! All pointers are device pointers.  `outoffsets` must have at least
//! `offsetslength` entries.

use std::os::raw::{c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_unique_strings_uint8(
        toptr: *mut u8,
        offsets: *const c_longlong,
        offsetslength: c_longlong,
        outoffsets: *mut c_longlong,
        out_count: *mut c_longlong, // device pointer
        stream: *mut c_void,
    );
}

/// Compact consecutive duplicate strings; report how many unique strings remain.
///
/// `out_count` is a device `i64` pointer; after the call it holds the number
/// of entries written to `outoffsets` (unique string count + 1).
pub fn hip_unique_strings_uint8(
    toptr: *mut u8,
    offsets: *const i64,
    offsetslength: i64,
    outoffsets: *mut i64,
    out_count: *mut i64,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_unique_strings_uint8(
            toptr,
            offsets as *const c_longlong,
            offsetslength as c_longlong,
            outoffsets as *mut c_longlong,
            out_count as *mut c_longlong,
            stream,
        );
    }
}
