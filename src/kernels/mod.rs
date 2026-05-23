// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Top‑level kernel module.
//!
//! This module exposes all compute backends:
//! - CPU (pure Rust kernels)
//! - SIMD (std::simd accelerated kernels)
//! - CUDA (GPU kernels via NVCC / Rust‑CUDA)
//! - HIP (GPU kernels for AMD via HIPCC)
//!
//! A dispatcher module may select the best backend at runtime.

pub mod cpu;
pub mod cuda;
#[cfg(all(feature = "hip", hip_rocm))]
pub mod hip;
pub mod simd;

pub mod dispatch;

// pub use dispatch::{backend::Backend, best_backend};

pub mod slice;
pub use slice::{Slice, SliceError, slice, slice_range};
