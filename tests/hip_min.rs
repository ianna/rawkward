// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#![cfg(all(feature = "hip", hip_rocm))]

#[link(name = "amdhip64")]
unsafe extern "C" {}

#[link(name = "awkward_hip_kernels", kind = "static")]
unsafe extern "C" {}

use rawkward::gpu::hip_available;
use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn hipFree(ptr: *mut c_void) -> i32;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
    fn hipDeviceSynchronize() -> i32;

    fn awkward_hip_segmented_min(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_void,
        n_segments: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

const H2D: i32 = 1;
const D2H: i32 = 2;

#[test]
fn test_hip_segmented_min() {
    if !hip_available() {
        eprintln!("Skipping HIP test: ROCm not available");
        return;
    }
    unsafe {
        let host_data = [5.0f32, 1.0, 9.0, 3.0];
        let host_offsets = [0i64, 2, 4];
        let mut host_out = vec![0f32; 2];

        let mut d_data: *mut c_void = std::ptr::null_mut();
        let mut d_offsets: *mut c_void = std::ptr::null_mut();
        let mut d_out: *mut c_void = std::ptr::null_mut();

        hipMalloc(&mut d_data, host_data.len() * 4);
        hipMalloc(&mut d_offsets, host_offsets.len() * 8);
        hipMalloc(&mut d_out, host_out.len() * 4);

        hipMemcpy(
            d_data,
            host_data.as_ptr() as *const c_void,
            host_data.len() * 4,
            H2D,
        );
        hipMemcpy(
            d_offsets,
            host_offsets.as_ptr() as *const c_void,
            host_offsets.len() * 8,
            H2D,
        );

        awkward_hip_segmented_min(
            d_data,
            d_offsets as *const i64,
            d_out,
            2,
            0, // float32
            std::ptr::null_mut(),
        );

        hipDeviceSynchronize();

        hipMemcpy(
            host_out.as_mut_ptr() as *mut c_void,
            d_out,
            host_out.len() * 4,
            D2H,
        );

        assert_eq!(host_out, vec![1.0, 3.0]);

        hipFree(d_data);
        hipFree(d_offsets);
        hipFree(d_out);
    }
}
