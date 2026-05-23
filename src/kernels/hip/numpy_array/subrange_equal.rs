// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP subrange-equal FFI wrapper.
//! Calls: awkward_hip_subrange_equal(tmpptr, fromstarts, fromstops, length,
//!                                   out_equal, dtype_code, stream)
//!
//! Checks whether all pairs of sub-ranges (defined by `fromstarts`/`fromstops`)
//! that have the same length compare element-wise equal.
//!
//! Writes `1` to `*out_equal` if all such pairs are equal, `0` otherwise.
//! `out_equal` is a device pointer to an `i32`.
//!
//! `dtype_code` selects the element type:
//!   0=i8, 1=u8, 2=i16, 3=u16, 4=i32, 5=u32, 6=i64, 7=u64, 8=f32, 9=f64

use std::os::raw::{c_int, c_longlong, c_void};

/// dtype codes for `subrange_equal`.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum SubrangeEqualDtype {
    I8 = 0,
    U8 = 1,
    I16 = 2,
    U16 = 3,
    I32 = 4,
    U32 = 5,
    I64 = 6,
    U64 = 7,
    F32 = 8,
    F64 = 9,
}

unsafe extern "C" {
    fn awkward_hip_subrange_equal(
        tmpptr: *const c_void,
        fromstarts: *const c_longlong,
        fromstops: *const c_longlong,
        length: c_longlong,
        out_equal: *mut c_int, // device pointer; 1=equal, 0=not equal
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

/// Test pairwise equality of same-length sub-ranges.
///
/// `out_equal` is a device `i32` pointer: `1` = all pairs equal, `0` = differ.
pub fn hip_subrange_equal<T>(
    tmpptr: *const T,
    fromstarts: *const i64,
    fromstops: *const i64,
    length: i64,
    out_equal: *mut i32,
    dtype: SubrangeEqualDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_subrange_equal(
            tmpptr as *const c_void,
            fromstarts as *const c_longlong,
            fromstops as *const c_longlong,
            length as c_longlong,
            out_equal as *mut c_int,
            dtype as c_int,
            stream,
        );
    }
}
