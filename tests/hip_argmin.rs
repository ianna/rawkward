// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#![cfg(all(feature = "hip", hip_rocm))]

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

    // Your kernel entry point
    fn awkward_hip_segmented_argmin(
        data: *const c_void,
        offsets: *const c_longlong,
        out: *mut c_longlong,
        n_segments: c_longlong,
        dtype_code: c_int,
        stream: *mut c_void,
    );
}

// HIP memcpy kinds
const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;

#[test]
fn test_hip_segmented_argmin() {
    if !hip_available() {
        eprintln!("Skipping HIP test: ROCm not available");
        return;
    }
    unsafe {
        // -----------------------------
        // Host data
        // -----------------------------
        let host_data = [3.0f32, 1.0, 5.0, 2.0];
        let host_offsets = [0i64, 2, 4]; // segments: [0:2], [2:4]
        let mut host_out = vec![-1i64; 2];

        // -----------------------------
        // Device allocations
        // -----------------------------
        let mut d_data: *mut c_void = std::ptr::null_mut();
        let mut d_offsets: *mut c_void = std::ptr::null_mut();
        let mut d_out: *mut c_void = std::ptr::null_mut();

        assert_eq!(hipMalloc(&mut d_data, host_data.len() * 4), 0);
        assert_eq!(hipMalloc(&mut d_offsets, host_offsets.len() * 8), 0);
        assert_eq!(hipMalloc(&mut d_out, host_out.len() * 8), 0);

        // -----------------------------
        // Copy host → device
        // -----------------------------
        assert_eq!(
            hipMemcpy(
                d_data,
                host_data.as_ptr() as *const c_void,
                host_data.len() * 4,
                HIP_MEMCPY_HOST_TO_DEVICE
            ),
            0
        );

        assert_eq!(
            hipMemcpy(
                d_offsets,
                host_offsets.as_ptr() as *const c_void,
                host_offsets.len() * 8,
                HIP_MEMCPY_HOST_TO_DEVICE
            ),
            0
        );

        // -----------------------------
        // Launch kernel
        // -----------------------------
        awkward_hip_segmented_argmin(
            d_data,
            d_offsets as *const i64,
            d_out as *mut i64,
            2,                    // n_segments
            0,                    // dtype_code = float32
            std::ptr::null_mut(), // default stream
        );

        assert_eq!(hipDeviceSynchronize(), 0);

        // -----------------------------
        // Copy device → host
        // -----------------------------
        assert_eq!(
            hipMemcpy(
                host_out.as_mut_ptr() as *mut c_void,
                d_out,
                host_out.len() * 8,
                HIP_MEMCPY_DEVICE_TO_HOST
            ),
            0
        );

        // -----------------------------
        // Validate results
        // -----------------------------
        assert_eq!(host_out, vec![1, 3]); // argmin([3,1]) = 1, argmin([5,2]) = 3

        // -----------------------------
        // Cleanup
        // -----------------------------
        hipFree(d_data);
        hipFree(d_offsets);
        hipFree(d_out);
    }
}
