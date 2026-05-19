// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::kernels::hip::reduce::HipDtype;
use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_segmented_argmax(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_longlong,
        n_segments: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

pub fn hip_segmented_argmax<T>(
    data: *const T,
    offsets: *const i64,
    out: *mut i64,
    n_segments: i64,
    dtype: HipDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_segmented_argmax(
            data as *const c_void,
            offsets as *const c_longlong,
            out as *mut c_longlong,
            n_segments as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
