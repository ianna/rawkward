// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP index-nones-as-index FFI wrapper.
//! Calls: awkward_hip_index_nones_as_index(toindex, length, stream)
//!
//! Replaces every -1 entry in `toindex` with a unique sequential value
//! that extends beyond the existing non-null entries.  Modified in place.
//!
//! Algorithm: count n_non_null, then replace -1s with n_non_null,
//! n_non_null+1, ... in left-to-right order.
//!
//! Corresponds to CPU: index_nones_as_index_64.

use std::os::raw::{c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_index_nones_as_index(
        toindex: *mut c_longlong,
        length: c_longlong,
        stream: *mut c_void,
    );
}

/// Replace -1 (None) entries with sequential indices beyond existing values.
///
/// `toindex` is modified in place on device.
pub fn hip_index_nones_as_index(toindex: *mut i64, length: i64, stream: *mut c_void) {
    unsafe {
        awkward_hip_index_nones_as_index(toindex as *mut c_longlong, length as c_longlong, stream);
    }
}
