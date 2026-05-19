// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP indexed-array range carry FFI wrappers.
//!
//! Two entry points:
//!
//! `awkward_hip_indexed_array_ranges_carry_next(tocarry, index, fromstarts,
//!     fromstops, nranges, out_count, dtype_code, stream)`
//!   For each range, append non-negative index values to tocarry.
//!   `out_count` (device i64) receives total entries written.
//!
//! `awkward_hip_indexed_array_ranges_next(index, fromstarts, fromstops,
//!     nranges, tostarts, tostops, out_total, dtype_code, stream)`
//!   Count non-null values per range → prefix-sum into tostarts/tostops.
//!   `out_total` (device i64) receives the grand total.
//!
//! `dtype_code`: 0=i32, 1=u32, 2=i64.

use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_indexed_array_ranges_carry_next(
        tocarry:    *mut c_longlong,
        index:      *const c_void,
        fromstarts: *const c_longlong,
        fromstops:  *const c_longlong,
        nranges:    c_longlong,
        out_count:  *mut c_longlong,
        dtype_code: c_int,
        stream:     *mut c_void,
    );

    fn awkward_hip_indexed_array_ranges_next(
        index:      *const c_void,
        fromstarts: *const c_longlong,
        fromstops:  *const c_longlong,
        nranges:    c_longlong,
        tostarts:   *mut c_longlong,
        tostops:    *mut c_longlong,
        out_total:  *mut c_longlong,
        dtype_code: c_int,
        stream:     *mut c_void,
    );
}

/// dtype code for range carry variants.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum RangesDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

/// Collect non-negative index values from each range into `tocarry`.
///
/// `out_count` (device i64) receives the total number of entries written.
pub fn hip_indexed_array_ranges_carry_next(
    tocarry:    *mut i64,
    index:      *const c_void,
    fromstarts: *const i64,
    fromstops:  *const i64,
    nranges:    i64,
    out_count:  *mut i64,
    dtype:      RangesDtype,
    stream:     *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_ranges_carry_next(
            tocarry as *mut c_longlong,
            index,
            fromstarts as *const c_longlong,
            fromstops  as *const c_longlong,
            nranges as c_longlong,
            out_count as *mut c_longlong,
            dtype as c_int,
            stream,
        );
    }
}

/// Compute tostarts/tostops from per-range non-null counts.
///
/// `out_total` (device i64) receives the grand total of non-null entries.
pub fn hip_indexed_array_ranges_next(
    index:      *const c_void,
    fromstarts: *const i64,
    fromstops:  *const i64,
    nranges:    i64,
    tostarts:   *mut i64,
    tostops:    *mut i64,
    out_total:  *mut i64,
    dtype:      RangesDtype,
    stream:     *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_ranges_next(
            index,
            fromstarts as *const c_longlong,
            fromstops  as *const c_longlong,
            nranges as c_longlong,
            tostarts  as *mut c_longlong,
            tostops   as *mut c_longlong,
            out_total as *mut c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
