// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! HIP indexed-array overlay-mask FFI wrapper.
//! Calls: awkward_hip_indexed_array_overlay_mask(toindex, mask, fromindex,
//!                                               length, dtype_code, stream)
//!
//! For each position i:
//!   mask[i] != 0  →  toindex[i] = -1
//!   mask[i] == 0  →  toindex[i] = fromindex[i]
//!
//! `dtype_code` selects the fromindex element type: 0=i32, 1=u32, 2=i64.
//! `toindex` is always i64.

use std::os::raw::{c_int, c_longlong, c_void};

/// dtype code for the `fromindex` array.
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum IndexDtype {
    I32 = 0,
    U32 = 1,
    I64 = 2,
}

unsafe extern "C" {
    fn awkward_hip_indexed_array_overlay_mask(
        toindex:    *mut c_longlong,
        mask:       *const i8,
        fromindex:  *const c_void,
        length:     c_longlong,
        dtype_code: c_int,
        stream:     *mut c_void,
    );
}

/// Apply a byte mask to an index array; masked positions become -1.
pub fn hip_indexed_array_overlay_mask(
    toindex:    *mut i64,
    mask:       *const i8,
    fromindex:  *const c_void,
    length:     i64,
    dtype:      IndexDtype,
    stream:     *mut c_void,
) {
    unsafe {
        awkward_hip_indexed_array_overlay_mask(
            toindex as *mut c_longlong,
            mask,
            fromindex,
            length as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
