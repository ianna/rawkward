// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP indexed-array fill FFI wrappers.
//!
//! Two entry points:
//!
//! `awkward_hip_indexed_array_fill(toindex, toindexoffset, fromindex, length, base, dtype_code, stream)`
//!   For each i: fromindex[i] < 0 → -1, else fromindex[i] + base.
//!   Written to toindex[toindexoffset + i].  dtype_code: 0=i32, 1=u32, 2=i64.
//!
//! `awkward_hip_indexed_array_fill_count(toindex, toindexoffset, length, base, stream)`
//!   For each i: toindex[toindexoffset + i] = base + i.

use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_indexed_array_fill(
        toindex: *mut c_longlong,
        toindexoffset: c_longlong,
        fromindex: *const c_void,
        length: c_longlong,
        base: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );

    fn awkward_hip_indexed_array_fill_count(
        toindex: *mut c_longlong,
        toindexoffset: c_longlong,
        length: c_longlong,
        base: c_longlong,
        stream: *mut c_void,
    );
}

/// dtype code for `hip_indexed_array_fill`.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum IndexFillDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

/// Fill a slice of `toindex` from `fromindex`, offsetting non-null values by `base`.
pub fn hip_indexed_array_fill(
    toindex: *mut i64,
    toindexoffset: i64,
    fromindex: *const c_void,
    length: i64,
    base: i64,
    dtype: IndexFillDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_fill(
            toindex as *mut c_longlong,
            toindexoffset as c_longlong,
            fromindex,
            length as c_longlong,
            base as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}

/// Fill `toindex[toindexoffset + i] = base + i` for `i` in `0..length`.
pub fn hip_indexed_array_fill_count(
    toindex: *mut i64,
    toindexoffset: i64,
    length: i64,
    base: i64,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_fill_count(
            toindex as *mut c_longlong,
            toindexoffset as c_longlong,
            length as c_longlong,
            base as c_longlong,
            stream,
        );
    }
}
