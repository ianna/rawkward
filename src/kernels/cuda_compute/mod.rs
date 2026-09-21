// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! CCCL/CUB compute path (option B).
//!
//! Where `kernels::cuda` (option A) launches hand-written per-segment kernels,
//! this module drives NVIDIA CUB's `DeviceSegmentedReduce` through thin
//! `extern "C"` host wrappers (`reduce.cu` + `cccl_common.cuh`). This is the
//! same strategy awkward-array uses for its CUDA reducers, which go through
//! `cuda.compute` (CCCL) rather than bespoke kernels — so it is the
//! apples-to-apples counterpart of awkward's CUDA backend.
//!
//! The whole module only exists when build.rs set `cfg(cuda_toolkit)` (nvcc +
//! driver present, `reduce.cu` compiled into a static lib and linked with
//! cudart). It is wired in under that cfg in `kernels/mod.rs`.

pub mod reduce;
