// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP rpad-and-clip index-building FFI wrappers.
//!
//! Three entry points:
//!
//! `awkward_hip_indexedoption_rpad_and_clip_mask_axis1(toindex, frommask,
//!     length, stream)`
//!   frommask[i]!=0 → toindex[i]=-1; else toindex[i]=count++.
//!
//! `awkward_hip_index_rpad_and_clip_axis0(toindex, target, length, stream)`
//!   toindex[0..shorter-1] = 0..shorter-1; toindex[shorter..target-1] = -1.
//!   shorter = min(target, length).
//!
//! `awkward_hip_index_rpad_and_clip_axis1(tostarts, tostops, target, length, stream)`
//!   tostarts[i] = i * target; tostops[i] = (i+1) * target.

use std::os::raw::{c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_indexedoption_rpad_and_clip_mask_axis1(
        toindex:  *mut c_longlong,
        frommask: *const i8,
        length:   c_longlong,
        stream:   *mut c_void,
    );

    fn awkward_hip_index_rpad_and_clip_axis0(
        toindex: *mut c_longlong,
        target:  c_longlong,
        length:  c_longlong,
        stream:  *mut c_void,
    );

    fn awkward_hip_index_rpad_and_clip_axis1(
        tostarts: *mut c_longlong,
        tostops:  *mut c_longlong,
        target:   c_longlong,
        length:   c_longlong,
        stream:   *mut c_void,
    );
}

/// Build toindex for rpad-and-clip on axis=1 of an IndexedOptionArray.
///
/// Masked positions → -1; unmasked positions get sequential indices.
pub fn hip_indexedoption_rpad_and_clip_mask_axis1(
    toindex:  *mut i64,
    frommask: *const i8,
    length:   i64,
    stream:   *mut c_void,
) {
    unsafe {
        awkward_hip_indexedoption_rpad_and_clip_mask_axis1(
            toindex as *mut c_longlong,
            frommask,
            length as c_longlong,
            stream,
        );
    }
}

/// Build toindex for rpad-and-clip on axis=0.
///
/// Fills `0..shorter-1` then `-1` to pad up to `target`.
/// `shorter = min(target, length)`.
pub fn hip_index_rpad_and_clip_axis0(
    toindex: *mut i64,
    target:  i64,
    length:  i64,
    stream:  *mut c_void,
) {
    unsafe {
        awkward_hip_index_rpad_and_clip_axis0(
            toindex as *mut c_longlong,
            target  as c_longlong,
            length  as c_longlong,
            stream,
        );
    }
}

/// Build tostarts/tostops for rpad-and-clip on axis=1.
///
/// `tostarts[i] = i * target`, `tostops[i] = (i+1) * target`.
pub fn hip_index_rpad_and_clip_axis1(
    tostarts: *mut i64,
    tostops:  *mut i64,
    target:   i64,
    length:   i64,
    stream:   *mut c_void,
) {
    unsafe {
        awkward_hip_index_rpad_and_clip_axis1(
            tostarts as *mut c_longlong,
            tostops  as *mut c_longlong,
            target   as c_longlong,
            length   as c_longlong,
            stream,
        );
    }
}
