// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP indexed-array flatten FFI wrappers.
//!
//! Two entry points:
//!
//! `awkward_hip_indexed_array_flatten_nextcarry(tocarry, fromindex, length,
//!     lencontent, out_count, out_error, dtype_code, stream)`
//!   Collect non-negative fromindex values into tocarry (skip -1s).
//!   `out_count` (device i64) receives number of entries written.
//!   Error if any non-negative entry >= lencontent.
//!
//! `awkward_hip_indexed_array_flatten_none2empty(outoffsets, outindex, offsets,
//!     outindexlength, offsetslength, out_error, dtype_code, stream)`
//!   Build outoffsets: None → zero length; valid → list length from offsets.
//!   Error if idx+1 >= offsetslength for a non-null entry.
//!
//! `dtype_code`: 0=i32, 1=u32, 2=i64.
//! `out_error`: device i32, caller zero-inits; 1 = error.

use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_indexed_array_flatten_nextcarry(
        tocarry:    *mut c_longlong,
        fromindex:  *const c_void,
        length:     c_longlong,
        lencontent: c_longlong,
        out_count:  *mut c_longlong,
        out_error:  *mut c_int,
        dtype_code: c_int,
        stream:     *mut c_void,
    );

    fn awkward_hip_indexed_array_flatten_none2empty(
        outoffsets:      *mut c_longlong,
        outindex:        *const c_void,
        offsets:         *const c_longlong,
        outindexlength:  c_longlong,
        offsetslength:   c_longlong,
        out_error:       *mut c_int,
        dtype_code:      c_int,
        stream:          *mut c_void,
    );
}

/// dtype code for flatten variants.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum FlattenDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

/// Collect non-negative fromindex entries into tocarry, skipping nulls.
///
/// `out_count` (device i64) receives the number of entries written.
pub fn hip_indexed_array_flatten_nextcarry(
    tocarry:    *mut i64,
    fromindex:  *const c_void,
    length:     i64,
    lencontent: i64,
    out_count:  *mut i64,
    out_error:  *mut i32,
    dtype:      FlattenDtype,
    stream:     *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_flatten_nextcarry(
            tocarry as *mut c_longlong,
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

/// Build output offsets for flatten-none-to-empty.
pub fn hip_indexed_array_flatten_none2empty(
    outoffsets:     *mut i64,
    outindex:       *const c_void,
    offsets:        *const i64,
    outindexlength: i64,
    offsetslength:  i64,
    out_error:      *mut i32,
    dtype:          FlattenDtype,
    stream:         *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_flatten_none2empty(
            outoffsets as *mut c_longlong,
            outindex,
            offsets as *const c_longlong,
            outindexlength as c_longlong,
            offsetslength as c_longlong,
            out_error as *mut c_int,
            dtype as c_int,
            stream,
        );
    }
}
