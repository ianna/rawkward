// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

/// Errors returned by GPU backends.
///
/// Implements [`std::error::Error`] and [`std::fmt::Display`] via `thiserror`,
/// so backend failures propagate as ordinary `Result`s instead of panicking.
#[derive(thiserror::Error, Debug)]
pub enum GpuError {
    #[error("HIP error: {0}")]
    HipError(String),
    #[error("CUDA error: {0}")]
    CudaError(String),
    #[error("Metal error: {0}")]
    MetalError(String),
    #[error("kernel not found: {0}")]
    KernelNotFound(String),
    #[error("invalid argument")]
    InvalidArgument,
    #[error("GPU initialization failed")]
    InitializationFailed,
}
