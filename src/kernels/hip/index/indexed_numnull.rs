// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP indexed-array null-counting FFI wrappers.
//!
//! Three entry points:
//!
//! `awkward_hip_indexed_array_numnull(fromindex, length, out_count, dtype_code, stream)`
//!   Count negative entries into `*out_count` (device i64, caller zero-inits).
//!
//! `awkward_hip_indexed_array_numnull_parents(numnull, fromindex, length, out_total, dtype_code, stream)`
//!   Per-element null flag (1/0) in `numnull[]`; total in `*out_total` (device i64).
//!
//! `awkward_hip_indexed_array_numnull_unique(toindex, lenindex, stream)`
//!   Fill `toindex[0..lenindex] = 0..lenindex-1`, `toindex[lenindex] = -1`.
//!
//! `dtype_code` for the first two: 0=i32, 1=u32, 2=i64.

use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_indexed_array_numnull(
        fromindex: *const c_void,
        length: c_longlong,
        out_count: *mut c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );

    fn awkward_hip_indexed_array_numnull_parents(
        numnull: *mut c_longlong,
        fromindex: *const c_void,
        length: c_longlong,
        out_total: *mut c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );

    fn awkward_hip_indexed_array_numnull_unique(
        toindex: *mut c_longlong,
        lenindex: c_longlong,
        stream: *mut c_void,
    );
}

/// dtype code for `hip_indexed_array_numnull` and `hip_indexed_array_numnull_parents`.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum NumnullDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

/// Count null (negative) entries in `fromindex`; result added into `*out_count`.
pub fn hip_indexed_array_numnull(
    fromindex: *const c_void,
    length: i64,
    out_count: *mut i64,
    dtype: NumnullDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_numnull(
            fromindex,
            length as c_longlong,
            out_count as *mut c_longlong,
            dtype as c_int,
            stream,
        );
    }
}

/// Write per-element null flag (1/0) to `numnull`; total added into `*out_total`.
pub fn hip_indexed_array_numnull_parents(
    numnull: *mut i64,
    fromindex: *const c_void,
    length: i64,
    out_total: *mut i64,
    dtype: NumnullDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_numnull_parents(
            numnull as *mut c_longlong,
            fromindex,
            length as c_longlong,
            out_total as *mut c_longlong,
            dtype as c_int,
            stream,
        );
    }
}

/// Fill `toindex[0..lenindex] = 0..lenindex-1`, `toindex[lenindex] = -1`.
pub fn hip_indexed_array_numnull_unique(toindex: *mut i64, lenindex: i64, stream: *mut c_void) {
    unsafe {
        awkward_hip_indexed_array_numnull_unique(
            toindex as *mut c_longlong,
            lenindex as c_longlong,
            stream,
        );
    }
}
