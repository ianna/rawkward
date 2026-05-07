// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use hip_sys::*;

extern "C" {
    fn hip_listoffset_carry_copy(
        from_offsets: *const i64,
        from_carry: *const i64,
        carry: *const i64,
        to_offsets: *mut i64,
        to_content: *mut i64,
        n_out_lists: i64,
    );
}

pub fn listoffset_carry(
    from_offsets: &DeviceBuffer<i64>,
    from_carry: &DeviceBuffer<i64>,
    carry: &DeviceBuffer<i64>,
    to_offsets: &DeviceBuffer<i64>,
    to_content: &DeviceBuffer<i64>,
    n_out_lists: usize,
    stream: hipStream_t,
) {
    unsafe {
        hip_listoffset_carry(
            from_offsets.as_device_ptr(),
            from_carry.as_device_ptr(),
            carry.as_device_ptr(),
            to_offsets.as_device_ptr(),
            to_content.as_device_ptr(),
            n_out_lists,
            stream,
        );
    }
}