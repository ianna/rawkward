// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! CUDA (NVIDIA) backend.
//!
//! `build.rs` emits `cfg(cuda_toolkit)` only when it detected the CUDA toolkit
//! (nvcc + the driver library) and compiled the kernels + generated bindings.
//! When present we use the real Driver-API backend; otherwise we fall back to
//! a stub whose `new()` returns an error, so the crate still builds on hosts
//! without CUDA (identical behavior to before this backend existed).

#[cfg(cuda_toolkit)]
mod imp_real;
#[cfg(cuda_toolkit)]
pub use imp_real::{CudaBackend, CudaKernelHandle, cuda_bindings};

#[cfg(not(cuda_toolkit))]
mod imp_stub;
#[cfg(not(cuda_toolkit))]
pub use imp_stub::{CudaBackend, CudaKernelHandle};
