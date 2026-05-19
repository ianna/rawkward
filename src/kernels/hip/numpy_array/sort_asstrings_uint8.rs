// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP sort-asstrings-uint8 FFI wrapper.
//! Calls: awkward_hip_sort_asstrings_uint8(toptr, fromptr, offsets,
//!                                         offsetslength, outoffsets,
//!                                         ascending, stable, stream)
//!
//! Sorts the strings defined by `fromptr` / `offsets` lexicographically,
//! writes sorted bytes to `toptr`, and writes new boundary positions to
//! `outoffsets[0..nstrings+1]`.
//!
//! `ascending = 1` for ascending order, `0` for descending.
//! `stable = 1` for stable sort, `0` for unstable (currently both use the
//! same insertion sort, which is inherently stable).
//!
//! All pointers are device pointers.  `outoffsets` must have at least
//! `offsetslength` entries.

use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_sort_asstrings_uint8(
        toptr:         *mut u8,
        fromptr:       *const u8,
        offsets:       *const c_longlong,
        offsetslength: c_longlong,
        outoffsets:    *mut c_longlong,
        ascending:     c_int,
        stable:        c_int,
        stream:        *mut c_void,
    );
}

/// Sort variable-length strings by lexicographic byte order.
///
/// `ascending`: `true` = smallest first, `false` = largest first.
/// `stable`: `true` = preserve relative order of equal strings.
pub fn hip_sort_asstrings_uint8(
    toptr:         *mut u8,
    fromptr:       *const u8,
    offsets:       *const i64,
    offsetslength: i64,
    outoffsets:    *mut i64,
    ascending:     bool,
    stable:        bool,
    stream:        *mut c_void,
) {
    unsafe {
        awkward_hip_sort_asstrings_uint8(
            toptr,
            fromptr,
            offsets as *const c_longlong,
            offsetslength as c_longlong,
            outoffsets as *mut c_longlong,
            ascending as c_int,
            stable as c_int,
            stream,
        );
    }
}
