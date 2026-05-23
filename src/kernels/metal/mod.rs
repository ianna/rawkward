// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Apple Silicon Metal GPU Compute Backend Kernels.
//!
//! This module mirrors the structural operations layout of the CPU, CUDA,
//! and HIP subsystems, managing the compilation and registration of
//! Metal Shading Language (MSL) kernels.

pub mod index;
pub mod macros;
pub mod numpy_array;
pub mod reduce;
pub mod sort;

use metal::{CompileOptions, ComputePipelineState, Device, Library};
use std::collections::HashMap;
use std::sync::Mutex;

/// Compiled pipeline state handle returned by `MetalKernelRegistry::get_pipeline`.
/// Wraps a `ComputePipelineState` so it can be passed through the generic
/// `GpuBackend::KernelHandle` associated type.
pub struct MetalKernelHandle(pub ComputePipelineState);

/// Thread-safe shader library manager for the Metal compute pipeline.
pub struct MetalKernelRegistry {
    library: Library,
    /// Cache compiled pipeline states to prevent expensive rebuilds during execution loops.
    pipelines: Mutex<HashMap<String, ComputePipelineState>>,
}

/// MSL source embedded into the binary at compile time.
/// No runtime file I/O or lazy initialisation needed — `include_str!` is a
/// compile-time constant.
const METAL_SHADER_SRC: &str = include_str!("kernels.metal");

impl MetalKernelRegistry {
    /// Instantiates the registry by compiling the embedded MSL source against
    /// the given device.  Returns an error string if compilation fails so the
    /// caller can surface it through `GpuError::MetalError`.
    pub fn new(device: &Device) -> Result<Self, String> {
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(METAL_SHADER_SRC, &options)
            .map_err(|err| format!("MSL compilation failed: {:?}", err))?;

        Ok(Self {
            library,
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    /// Returns a cached `ComputePipelineState`, building it on first use.
    pub fn get_pipeline(
        &self,
        device: &Device,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        let mut cache = self.pipelines.lock().unwrap();

        if let Some(pipeline) = cache.get(name) {
            return Ok(pipeline.clone());
        }

        // In metal 0.29, `get_function` returns `Result<Function, E>` rather
        // than `Option<Function>`, so use `map_err` to convert the error.
        let function = self
            .library
            .get_function(name, None)
            .map_err(|err| format!("Metal kernel '{}' not found: {:?}", name, err))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|err| format!("Pipeline creation failed for '{}': {:?}", name, err))?;

        cache.insert(name.to_string(), pipeline.clone());
        Ok(pipeline)
    }
}
