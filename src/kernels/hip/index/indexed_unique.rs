// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP indexed-array unique-next FFI wrapper.
//! Calls: awkward_hip_indexed_array_unique_next_index_and_offsets(
//!            toindex, tooffsets, fromoffsets, fromnulls, startslength, stream)
//!
//! Fills `toindex` and `tooffsets` for the unique-values reduction step.
//!
//! For each group i (defined by fromoffsets[i..i+1]):
//!   Each element gets a sequential index; if fromnulls[k]==1 after the group,
//!   a -1 sentinel is inserted and the group boundary is extended by 1.

use std::os::raw::{c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_indexed_array_unique_next_index_and_offsets(
        toindex: *mut c_longlong,
        tooffsets: *mut c_longlong,
        fromoffsets: *const c_longlong,
        fromnulls: *const c_longlong,
        startslength: c_longlong,
        stream: *mut c_void,
    );
}

/// Build toindex and tooffsets for the next unique-values step.
pub fn hip_indexed_array_unique_next_index_and_offsets(
    toindex: *mut i64,
    tooffsets: *mut i64,
    fromoffsets: *const i64,
    fromnulls: *const i64,
    startslength: i64,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_unique_next_index_and_offsets(
            toindex as *mut c_longlong,
            tooffsets as *mut c_longlong,
            fromoffsets as *const c_longlong,
            fromnulls as *const c_longlong,
            startslength as c_longlong,
            stream,
        );
    }
}
