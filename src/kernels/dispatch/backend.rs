// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Backend selection and dispatch logic.
//!
//! Priority order:
//!   1. Metal (Apple Silicon GPU — macOS only)
//!   2. CUDA  (NVIDIA GPU)
//!   3. HIP   (AMD GPU)
//!   4. SIMD  (CPU vectorization)
//!   5. CPU   (scalar fallback)

#![allow(dead_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Simd,
    Cuda,
    Hip,
    #[cfg(target_os = "macos")]
    Metal,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Simd => "simd",
            Backend::Cuda => "cuda",
            Backend::Hip => "hip",
            #[cfg(target_os = "macos")]
            Backend::Metal => "metal",
        }
    }
}
