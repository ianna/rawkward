//! Error type used by every kernel in this crate.
//!
//! The original C++ kernels return an opaque `ERROR` value produced by either
//! `success()` or `failure(message, id, …)`.  Here we replace that with a
//! typed `Result<(), KernelError>`, giving callers structured error
//! information without any heap allocation on the happy path.

// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use std::fmt;

/// Sentinel value used by the C++ kernels for "no slice index" (`kSliceNone`).
pub const SLICE_NONE: i64 = i64::MIN;

/// Error returned by a kernel when its input data fails a validity check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelError {
    /// Human-readable description of what went wrong (mirrors the C++ message
    /// strings like `"start[i] > stop[i]"`).
    pub message: &'static str,
    /// The element index at which the violation was detected.
    pub id: i64,
    /// Optional slice index, or [`SLICE_NONE`] when not applicable.
    pub slice: i64,
}

impl KernelError {
    /// Construct a new error.
    #[inline]
    pub fn new(message: &'static str, id: i64, slice: i64) -> Self {
        Self { message, id, slice }
    }

    /// Convenience constructor that sets `slice` to [`SLICE_NONE`].
    #[inline]
    pub fn at(message: &'static str, id: i64) -> Self {
        Self::new(message, id, SLICE_NONE)
    }
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KernelError: {} (at index {})", self.message, self.id)
    }
}

impl std::error::Error for KernelError {}
