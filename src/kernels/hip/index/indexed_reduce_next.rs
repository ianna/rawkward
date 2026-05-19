// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP indexed-array reduce-next FFI wrappers.
//!
//! Four entry points:
//!
//! `awkward_hip_indexed_array_reduce_next(nextcarry, nextparents, outindex,
//!     index, parents, length, out_count, dtype_code, stream)`
//!   Build nextcarry/nextparents/outindex for a reduction step.
//!   `out_count` (device i64) receives the number of valid entries.
//!
//! `awkward_hip_indexed_array_reduce_next_fix_offsets(outoffsets, starts,
//!     startslength, outindexlength, stream)`
//!   Copy starts into outoffsets and append outindexlength as the sentinel.
//!
//! `awkward_hip_indexed_array_reduce_next_nonlocal_nextshifts(nextshifts,
//!     index, length, out_count, dtype_code, stream)`
//!   Cumulative null count for each non-null position.
//!
//! `awkward_hip_indexed_array_reduce_next_nonlocal_nextshifts_fromshifts(
//!     nextshifts, index, shifts, length, out_count, dtype_code, stream)`
//!   Like above but adds incoming shifts[i] for each non-null entry.
//!
//! `dtype_code`: 0=i32, 1=u32, 2=i64.

use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_indexed_array_reduce_next(
        nextcarry:    *mut c_longlong,
        nextparents:  *mut c_longlong,
        outindex:     *mut c_longlong,
        index:        *const c_void,
        parents:      *const c_longlong,
        length:       c_longlong,
        out_count:    *mut c_longlong,
        dtype_code:   c_int,
        stream:       *mut c_void,
    );

    fn awkward_hip_indexed_array_reduce_next_fix_offsets(
        outoffsets:      *mut c_longlong,
        starts:          *const c_longlong,
        startslength:    c_longlong,
        outindexlength:  c_longlong,
        stream:          *mut c_void,
    );

    fn awkward_hip_indexed_array_reduce_next_nonlocal_nextshifts(
        nextshifts: *mut c_longlong,
        index:      *const c_void,
        length:     c_longlong,
        out_count:  *mut c_longlong,
        dtype_code: c_int,
        stream:     *mut c_void,
    );

    fn awkward_hip_indexed_array_reduce_next_nonlocal_nextshifts_fromshifts(
        nextshifts: *mut c_longlong,
        index:      *const c_void,
        shifts:     *const c_longlong,
        length:     c_longlong,
        out_count:  *mut c_longlong,
        dtype_code: c_int,
        stream:     *mut c_void,
    );
}

/// dtype code for reduce-next variants.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum ReduceNextDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

/// Build nextcarry/nextparents/outindex for a reduction step.
///
/// `out_count` (device i64) receives the number of valid (non-null) entries.
pub fn hip_indexed_array_reduce_next(
    nextcarry:   *mut i64,
    nextparents: *mut i64,
    outindex:    *mut i64,
    index:       *const c_void,
    parents:     *const i64,
    length:      i64,
    out_count:   *mut i64,
    dtype:       ReduceNextDtype,
    stream:      *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_reduce_next(
            nextcarry   as *mut c_longlong,
            nextparents as *mut c_longlong,
            outindex    as *mut c_longlong,
            index,
            parents as *const c_longlong,
            length as c_longlong,
            out_count as *mut c_longlong,
            dtype as c_int,
            stream,
        );
    }
}

/// Copy `starts` into `outoffsets` and append `outindexlength` as the final entry.
pub fn hip_indexed_array_reduce_next_fix_offsets(
    outoffsets:     *mut i64,
    starts:         *const i64,
    startslength:   i64,
    outindexlength: i64,
    stream:         *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_reduce_next_fix_offsets(
            outoffsets as *mut c_longlong,
            starts     as *const c_longlong,
            startslength   as c_longlong,
            outindexlength as c_longlong,
            stream,
        );
    }
}

/// Cumulative null count for each non-null position.
///
/// `out_count` (device i64) receives the number of non-null entries written.
pub fn hip_indexed_array_reduce_next_nonlocal_nextshifts(
    nextshifts: *mut i64,
    index:      *const c_void,
    length:     i64,
    out_count:  *mut i64,
    dtype:      ReduceNextDtype,
    stream:     *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_reduce_next_nonlocal_nextshifts(
            nextshifts as *mut c_longlong,
            index,
            length as c_longlong,
            out_count as *mut c_longlong,
            dtype as c_int,
            stream,
        );
    }
}

/// Cumulative null count plus incoming shifts for each non-null position.
///
/// `out_count` (device i64) receives the number of non-null entries written.
pub fn hip_indexed_array_reduce_next_nonlocal_nextshifts_fromshifts(
    nextshifts: *mut i64,
    index:      *const c_void,
    shifts:     *const i64,
    length:     i64,
    out_count:  *mut i64,
    dtype:      ReduceNextDtype,
    stream:     *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_reduce_next_nonlocal_nextshifts_fromshifts(
            nextshifts as *mut c_longlong,
            index,
            shifts as *const c_longlong,
            length as c_longlong,
            out_count as *mut c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
