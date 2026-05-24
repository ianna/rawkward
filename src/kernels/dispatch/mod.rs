// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

pub mod backend;
pub mod device_query;

use backend::Backend;
#[cfg(target_os = "macos")]
use device_query::metal_available;
use device_query::{cuda_available, hip_available, simd_available};

/// Select the best backend available on this machine.
///
/// Priority: Metal (macOS) > CUDA > HIP > SIMD > CPU.
pub fn best_backend() -> Backend {
    // Metal check is compiled away entirely on non-macOS targets so the
    // `Backend::Metal` variant (also cfg-gated) never appears elsewhere.
    #[cfg(target_os = "macos")]
    if metal_available() {
        return Backend::Metal;
    }

    if cuda_available() {
        Backend::Cuda
    } else if hip_available() {
        Backend::Hip
    } else if simd_available() {
        Backend::Simd
    } else {
        Backend::Cpu
    }
}
