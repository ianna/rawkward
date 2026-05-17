// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#[derive(Debug)]
pub enum GpuError {
    HipError(String),
    CudaError(String),
    KernelNotFound(String),
    InvalidArgument,
}
