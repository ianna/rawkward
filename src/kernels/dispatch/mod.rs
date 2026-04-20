// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#![allow(dead_code)]

pub mod backend;
pub mod device_query;

use backend::Backend;
use device_query::{cuda_available, hip_available, simd_available};

/// Select the best backend available on this machine.
pub fn best_backend() -> Backend {
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
