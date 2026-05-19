// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP reduce-mask-bytemasked FFI wrapper.
//! Calls: awkward_hip_reduce_mask_bytemasked(toptr, parents, ngroups, nparents, stream)
//!
//! Builds a `ByteMaskedArray` validity mask:
//! - Initialises `toptr[0..ngroups]` to `1` (null / masked).
//! - Sets `toptr[parents[j]]` to `0` (valid) for each entry in `parents`.
//!
//! `toptr` must have `ngroups` bytes allocated on the device.

use std::os::raw::{c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_reduce_mask_bytemasked(
        toptr:    *mut i8,
        parents:  *const c_longlong,
        ngroups:  c_longlong,
        nparents: c_longlong,
        stream:   *mut c_void,
    );
}

/// Write a byte-masked validity array.
///
/// `toptr[g] = 1` (null) for every group; then `toptr[parents[j]] = 0` (valid).
pub fn hip_reduce_mask_bytemasked(
    toptr:    *mut i8,
    parents:  *const i64,
    ngroups:  i64,
    nparents: i64,
    stream:   *mut c_void,
) {
    unsafe {
        awkward_hip_reduce_mask_bytemasked(
            toptr,
            parents as *const c_longlong,
            ngroups as c_longlong,
            nparents as c_longlong,
            stream,
        );
    }
}
