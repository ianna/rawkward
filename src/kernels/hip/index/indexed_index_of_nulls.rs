// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP indexed-array index-of-nulls FFI wrapper.
//! Calls: awkward_hip_indexed_array_index_of_nulls(toindex, fromindex, parents,
//!                                                  starts, length, out_count,
//!                                                  dtype_code, stream)
//!
//! For each null entry (fromindex[i] < 0), records its within-list position:
//!   toindex[j++] = i - starts[parents[i]]
//!
//! `out_count` (device i64) receives the number of nulls written.
//! `dtype_code`: 0=i32, 1=u32, 2=i64.

use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_indexed_array_index_of_nulls(
        toindex: *mut c_longlong,
        fromindex: *const c_void,
        parents: *const c_longlong,
        starts: *const c_longlong,
        length: c_longlong,
        out_count: *mut c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

/// dtype code for `hip_indexed_array_index_of_nulls`.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum IndexOfNullsDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

/// Collect within-list positions of null entries into `toindex`.
///
/// `out_count` (device i64) receives the number of nulls written.
pub fn hip_indexed_array_index_of_nulls(
    toindex: *mut i64,
    fromindex: *const c_void,
    parents: *const i64,
    starts: *const i64,
    length: i64,
    out_count: *mut i64,
    dtype: IndexOfNullsDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_index_of_nulls(
            toindex as *mut c_longlong,
            fromindex,
            parents as *const c_longlong,
            starts as *const c_longlong,
            length as c_longlong,
            out_count as *mut c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
