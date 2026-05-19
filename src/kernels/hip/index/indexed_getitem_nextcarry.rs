// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP indexed-array getitem-nextcarry FFI wrappers.
//!
//! Two entry points:
//!
//! `awkward_hip_indexed_array_getitem_nextcarry(tocarry, fromindex, length,
//!     lencontent, out_error, dtype_code, stream)`
//!   Copy fromindex to tocarry; error if any entry is < 0 or >= lencontent.
//!
//! `awkward_hip_indexed_array_getitem_nextcarry_outindex(tocarry, toindex,
//!     fromindex, length, lencontent, out_count, out_error, dtype_code, stream)`
//!   Split: null entries → toindex[i]=-1, valid → tocarry[k]=j, toindex[i]=k.
//!   `out_count` (device i64) receives the number of entries written to tocarry.
//!
//! `dtype_code`: 0=i32, 1=u32, 2=i64.
//! `out_error`: device i32, caller zero-inits; 1 = out-of-range error.

use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_indexed_array_getitem_nextcarry(
        tocarry: *mut c_longlong,
        fromindex: *const c_void,
        length: c_longlong,
        lencontent: c_longlong,
        out_error: *mut c_int,
        dtype_code: c_int,
        stream: *mut c_void,
    );

    fn awkward_hip_indexed_array_getitem_nextcarry_outindex(
        tocarry: *mut c_longlong,
        toindex: *mut c_longlong,
        fromindex: *const c_void,
        length: c_longlong,
        lencontent: c_longlong,
        out_count: *mut c_longlong,
        out_error: *mut c_int,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

/// dtype code for getitem_nextcarry variants.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum GetitemCarryDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

/// Copy all fromindex entries to tocarry; error if out of range.
pub fn hip_indexed_array_getitem_nextcarry(
    tocarry: *mut i64,
    fromindex: *const c_void,
    length: i64,
    lencontent: i64,
    out_error: *mut i32,
    dtype: GetitemCarryDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_getitem_nextcarry(
            tocarry as *mut c_longlong,
            fromindex,
            length as c_longlong,
            lencontent as c_longlong,
            out_error as *mut c_int,
            dtype as c_int,
            stream,
        );
    }
}

/// Split fromindex into carry (valid entries) and output index (with -1 for nulls).
///
/// `out_count` (device i64) receives the number of entries written to `tocarry`.
pub fn hip_indexed_array_getitem_nextcarry_outindex(
    tocarry: *mut i64,
    toindex: *mut i64,
    fromindex: *const c_void,
    length: i64,
    lencontent: i64,
    out_count: *mut i64,
    out_error: *mut i32,
    dtype: GetitemCarryDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_getitem_nextcarry_outindex(
            tocarry as *mut c_longlong,
            toindex as *mut c_longlong,
            fromindex,
            length as c_longlong,
            lencontent as c_longlong,
            out_count as *mut c_longlong,
            out_error as *mut c_int,
            dtype as c_int,
            stream,
        );
    }
}
