// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Link HIP runtime
#[link(name = "amdhip64")]
unsafe extern "C" {}

// Link your static HIP kernels
#[link(name = "awkward_hip_kernels", kind = "static")]
unsafe extern "C" {}

use rawkward::gpu::hip_available;
use std::os::raw::{c_int, c_longlong, c_void};

unsafe extern "C" {
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn hipFree(ptr: *mut c_void) -> i32;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
    fn hipDeviceSynchronize() -> i32;

    fn awkward_hip_segmented_argmax(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_longlong,
        n_segments: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

const H2D: i32 = 1;
const D2H: i32 = 2;

#[test]
fn test_hip_segmented_argmax() {
    if !hip_available() {
        eprintln!("Skipping HIP test: ROCm not available");
        return;
    }
    unsafe {
        // Two segments: [0:2] and [2:4]
        let host_data = vec![3.0f32, 7.0, 1.0, 9.0];
        let host_offsets = vec![0i64, 2, 4];
        let mut host_out = vec![-1i64; 2];

        let mut d_data: *mut c_void = std::ptr::null_mut();
        let mut d_offsets: *mut c_void = std::ptr::null_mut();
        let mut d_out: *mut c_void = std::ptr::null_mut();

        hipMalloc(&mut d_data, host_data.len() * 4);
        hipMalloc(&mut d_offsets, host_offsets.len() * 8);
        hipMalloc(&mut d_out, host_out.len() * 8);

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

        awkward_hip_segmented_argmax(
            d_data,
            d_offsets as *const i64,
            d_out as *mut i64,
            2, // n_segments
            0, // dtype_code = float32
            std::ptr::null_mut(),
        );

        hipDeviceSynchronize();

        hipMemcpy(
            host_out.as_mut_ptr() as *mut c_void,
            d_out,
            host_out.len() * 8,
            D2H,
        );

        // Segment 0: [3,7] → argmax = index 1
        // Segment 1: [1,9] → argmax = index 3
        assert_eq!(host_out, vec![1, 3]);

        hipFree(d_data);
        hipFree(d_offsets);
        hipFree(d_out);
    }
}
