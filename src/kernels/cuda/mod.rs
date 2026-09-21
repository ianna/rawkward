// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! CUDA (NVIDIA) kernels.
//!
//! This module mirrors the structural operations layout of the CPU, HIP and
//! Metal backends. Kernel *sources* live in the sibling `.cu` files (compiled
//! by `build.rs` into a fatbin when the CUDA toolkit is present); the Rust
//! dispatch functions here are generic over `GpuBackend` and launch kernels by
//! name via `cuLaunchKernel`, exactly like the HIP dispatch.

pub mod macros;
pub mod reduce;
