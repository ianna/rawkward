// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

pub mod cuda;
pub mod device_slice;
pub mod error;
pub mod hip;
pub mod stream;
pub mod traits;

pub use device_slice::{DevSlice, DevicePtr};
pub use stream::GpuStream;
pub use traits::GpuBackend;

pub use cuda::CudaBackend;
pub use error::GpuError;
pub use hip::HipBackend;

pub enum BackendKind {
    Hip,
    Cuda,
}

pub enum BackendInner {
    Hip(HipBackend),
    Cuda(CudaBackend),
}

pub struct Backend {
    pub kind: BackendKind,
    pub inner: BackendInner,
}

impl Backend {
    pub fn new(kind: BackendKind) -> Result<Self, GpuError> {
        match kind {
            BackendKind::Hip => Ok(Self {
                kind,
                inner: BackendInner::Hip(HipBackend::new()),
            }),
            BackendKind::Cuda => Ok(Self {
                kind,
                inner: BackendInner::Cuda(CudaBackend::new()?),
            }),
        }
    }
}
