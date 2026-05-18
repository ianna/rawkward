// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#[cfg(feature = "hip")]
use crate::backend::hip::hip_bindings::hipStream_t;

pub enum GpuStream {
    #[cfg(feature = "hip")]
    Hip(hipStream_t),
    Cpu,
    Simd,
}
