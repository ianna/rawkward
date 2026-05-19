// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP indexed-array validity FFI wrapper.
//! Calls: awkward_hip_indexed_array_validity(index, length, lencontent,
//!                                           isoption, out_error, stream)
//!
//! Validates an IndexedArray or IndexedOptionArray index against a content length.
//! Rules:
//! * Non-option arrays (`isoption = false`): every index must be >= 0.
//! * Option arrays (`isoption = true`): -1 is allowed; non-negative values
//!   must be < lencontent.
//!
//! `out_error` is a device `i32` pointer (caller zero-inits):
//!   0 = valid, 1 = negative index in non-option, 2 = index out of range.

use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_indexed_array_validity(
        index:      *const c_longlong,
        length:     c_longlong,
        lencontent: c_longlong,
        isoption:   c_int,
        out_error:  *mut c_int,
        stream:     *mut c_void,
    );
}

/// Validate an IndexedArray index on device.
///
/// `out_error` is a device `i32` pointer: `0` = valid, `1` = negative-index
/// error (non-option), `2` = out-of-range error.
pub fn hip_indexed_array_validity(
    index:      *const i64,
    length:     i64,
    lencontent: i64,
    isoption:   bool,
    out_error:  *mut i32,
    stream:     *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_validity(
            index as *const c_longlong,
            length as c_longlong,
            lencontent as c_longlong,
            isoption as c_int,
            out_error as *mut c_int,
            stream,
        );
    }
}
