// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP segmented sort FFI wrapper.
//! Calls: awkward_hip_segmented_sort(data, offsets, out, n_segments, parentslength,
//!                                   dtype_code, ascending, stable, stream)
//!
//! Sorts the values within each segment using a three-tier batched bitonic
//! network (≤64 / ≤256 / ≤4096 elements).  The output buffer `out` has the
//! same length as `data`; `out[offsets[seg] .. offsets[seg+1]]` contains the
//! sorted values of segment `seg`.
//!
//! `parentslength` gates write-back: only elements at flat index < parentslength
//! are written to `out`.  Pass the total element count for an unrestricted sort.
//!
//! `ascending`: true = ascending order, false = descending.
//! `stable`: accepted but ignored (bitonic networks are inherently unstable).
//! NaN sorts first when ascending, last when descending (matches CPU ArgsortOrd).
//!
//! Lists longer than 4096 elements are silently left unchanged.

use std::os::raw::{c_int, c_longlong, c_void};

/// Element dtype passed to `awkward_hip_segmented_sort`.
///
/// Values match `dtype_code` in `sort.hip.cpp`:
/// Bool=0, I8=1, U8=2, I16=3, U16=4, I32=5, U32=6, I64=7, U64=8, F32=9, F64=10.
#[repr(i32)]
pub enum SortDtype {
    Bool = 0,
    I8 = 1,
    U8 = 2,
    I16 = 3,
    U16 = 4,
    I32 = 5,
    U32 = 6,
    I64 = 7,
    U64 = 8,
    F32 = 9,
    F64 = 10,
}

unsafe extern "C" {
    fn awkward_hip_segmented_sort(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_void,
        n_segments: c_longlong,
        parentslength: c_longlong,
        dtype_code: c_int,
        ascending: c_int,
        stable: c_int,
        stream: *mut c_void,
    );
}

/// Sort the values within each segment.
///
/// `out` must have the same number of elements as `data` (i.e. `offsets[n_segments]`).
/// `out[offsets[seg] .. offsets[seg+1]]` holds the sorted values of segment `seg`.
///
/// Only flat indices `< parentslength` are written; pass the total element count
/// for an unrestricted sort.
pub fn hip_segmented_sort<T>(
    data: *const T,
    offsets: *const i64,
    out: *mut T,
    n_segments: i64,
    parentslength: i64,
    dtype: SortDtype,
    ascending: bool,
    stable: bool,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_segmented_sort(
            data as *const c_void,
            offsets as *const c_longlong,
            out as *mut c_void,
            n_segments as c_longlong,
            parentslength as c_longlong,
            dtype as c_int,
            ascending as c_int,
            stable as c_int,
            stream,
        );
    }
}
