// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

pub mod cuda;
pub mod device_slice;
pub mod error;
#[cfg(all(feature = "hip", hip_rocm))]
pub mod hip;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod stream;
pub mod traits;

pub use device_slice::{DevSlice, DevicePtr};
pub use stream::GpuStream;
pub use traits::GpuBackend;

pub use cuda::CudaBackend;
pub use error::GpuError;
#[cfg(all(feature = "hip", hip_rocm))]
pub use hip::HipBackend;
#[cfg(target_os = "macos")]
pub use metal::MetalBackend;

pub enum BackendKind {
    #[cfg(all(feature = "hip", hip_rocm))]
    Hip,
    Cuda,
    #[cfg(target_os = "macos")]
    Metal,
}

pub enum BackendInner {
    #[cfg(all(feature = "hip", hip_rocm))]
    Hip(HipBackend),
    Cuda(CudaBackend),
    #[cfg(target_os = "macos")]
    Metal(MetalBackend),
}

pub struct Backend {
    pub kind: BackendKind,
    pub inner: BackendInner,
}

impl Backend {
    pub fn new(kind: BackendKind) -> Result<Self, GpuError> {
        match kind {
            #[cfg(all(feature = "hip", hip_rocm))]
            BackendKind::Hip => Ok(Self {
                kind,
                inner: BackendInner::Hip(HipBackend::new()?),
            }),
            BackendKind::Cuda => Ok(Self {
                kind,
                inner: BackendInner::Cuda(CudaBackend::new()?),
            }),
            #[cfg(target_os = "macos")]
            BackendKind::Metal => Ok(Self {
                kind,
                inner: BackendInner::Metal(MetalBackend::new()?),
            }),
        }
    }
}
