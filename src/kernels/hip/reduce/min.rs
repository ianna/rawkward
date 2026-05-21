// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::kernels::hip::reduce::HipDtype;
use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn awkward_hip_segmented_min(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_void,
        n_segments: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

pub fn hip_segmented_min<T>(
    data: *const T,
    offsets: *const i64,
    out: *mut T,
    n_segments: i64,
    dtype: HipDtype,
    stream: *mut c_void,
) {
    unsafe {
        awkward_hip_segmented_min(
            data as *const c_void,
            offsets as *const c_longlong,
            out as *mut c_void,
            n_segments as c_longlong,
            dtype as c_int,
            stream,
        );
    }
}
