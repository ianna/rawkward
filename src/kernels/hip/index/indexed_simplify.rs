// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP indexed-array simplify FFI wrapper.
//! Calls: awkward_hip_indexed_array_simplify(toindex, outerindex, innerindex,
//!                                           outerlength, innerlength,
//!                                           outer_dtype, inner_dtype,
//!                                           out_error, stream)
//!
//! Composes two index arrays: toindex[i] = innerindex[outerindex[i]],
//! or -1 if outerindex[i] < 0, or error if outerindex[i] >= innerlength.
//!
//! `outer_dtype` / `inner_dtype`: 0=i32, 1=u32, 2=i64.
//! `out_error` is a device `i32` (caller zero-inits); 1 = out-of-range.

use std::os::raw::{c_int, c_longlong, c_void};

/// dtype code for outer/inner index arrays.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum SimplifyDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_indexed_array_simplify(
        toindex:      *mut c_longlong,
        outerindex:   *const c_void,
        innerindex:   *const c_void,
        outerlength:  c_longlong,
        innerlength:  c_longlong,
        outer_dtype:  c_int,
        inner_dtype:  c_int,
        out_error:    *mut c_int,
        stream:       *mut c_void,
    );
}

/// Compose two index arrays: `toindex[i] = innerindex[outerindex[i]]`.
///
/// `out_error` is a device `i32` pointer: `0` = ok, `1` = out-of-range.
pub fn hip_indexed_array_simplify(
    toindex:     *mut i64,
    outerindex:  *const c_void,
    innerindex:  *const c_void,
    outerlength: i64,
    innerlength: i64,
    outer_dtype: SimplifyDtype,
    inner_dtype: SimplifyDtype,
    out_error:   *mut i32,
    stream:      *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_simplify(
            toindex as *mut c_longlong,
            outerindex,
            innerindex,
            outerlength as c_longlong,
            innerlength as c_longlong,
            outer_dtype as c_int,
            inner_dtype as c_int,
            out_error as *mut c_int,
            stream,
        );
    }
}
