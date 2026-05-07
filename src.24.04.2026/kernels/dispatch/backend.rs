// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Backend selection and dispatch logic.
//!
//! Priority order:
//!   1. CUDA (NVIDIA GPU)
//!   2. HIP  (AMD GPU)
//!   3. SIMD (CPU vectorization)
//!   4. CPU  (fallback)

#![allow(dead_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Simd,
    Cuda,
    Hip,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Simd => "simd",
            Backend::Cuda => "cuda",
            Backend::Hip => "hip",
        }
    }
}
