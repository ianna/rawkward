// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#[cfg(all(feature = "hip", hip_rocm))]
use crate::backend::hip::hip_bindings::hipStream_t;

#[cfg(cuda_toolkit)]
use crate::backend::cuda::cuda_bindings::CUstream;

pub enum GpuStream {
    #[cfg(all(feature = "hip", hip_rocm))]
    Hip(hipStream_t),

    #[cfg(cuda_toolkit)]
    Cuda(CUstream),

    #[cfg(target_os = "macos")]
    Metal(metal::CommandQueue),

    Cpu,
    Simd,
}
