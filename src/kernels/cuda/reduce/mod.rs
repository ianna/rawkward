// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// `CudaDtype` is the shared dtype tag used by all reduce modules.  It lives in
// `argmin` (its first consumer) and is re-exported here so every sibling can
// import it as `crate::kernels::cuda::reduce::CudaDtype`.
//
// These dispatch functions are generic over `B: GpuBackend`; they are the CUDA
// counterparts of `src/kernels/hip/reduce/*`, launching kernels of the same
// names (`segmented_sum_f32`, …) compiled from the sibling `.cu` sources.
pub use argmin::CudaDtype;

// Scalar reducers — output has one element per segment
pub mod argmax;
pub mod argmin;
pub mod count;
pub mod countnonzero;
pub mod max;
pub mod min;
pub mod prod;
pub mod sum;
