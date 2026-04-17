// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Hardware capability detection for backend selection.

#![allow(dead_code)]

pub fn cuda_available() -> bool {
    // TODO: call into CUDA driver API (cuInit, cuDeviceGetCount)
    false
}

pub fn hip_available() -> bool {
    // TODO: call into HIP runtime (hipInit, hipGetDeviceCount)
    false
}

pub fn simd_available() -> bool {
    // TODO: detect AVX2/AVX512/SVE/etc.
    // For now, assume SIMD is always available.
    true
}

