// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! # cpu-kernels
//!
//! Pure-Rust implementations of the Awkward Array CPU kernels originally
//! written in C++ (`awkward/src/cpu-kernels/awkward_*.cpp`).
//!
//! Each subdirectory groups all kernels for one Awkward layout type:
//!
//! | Subdir              | Covers C++ files prefixed with                                       |
//! |---------------------|----------------------------------------------------------------------|
//! | [`mask`]            | `awkward_BitMaskedArray_*`, `awkward_ByteMaskedArray_*`              |
//! | [`content`]         | `awkward_Content_*`                                                  |
//! | [`index`]           | `awkward_Index_*`, `awkward_IndexedArray_*`, `awkward_IndexedOptionArray_*` |
//! | [`list_array`]      | `awkward_ListArray_*`                                                |
//! | [`list_offset`]     | `awkward_ListOffsetArray_*`                                          |
//! | [`masked_array`]    | `awkward_MaskedArray_*`                                              |
//! | [`numpy_array`]     | `awkward_NumpyArray_*`                                               |
//! | [`record_array`]    | `awkward_RecordArray_*`                                              |
//! | [`regular_array`]   | `awkward_RegularArray_*`                                             |
//! | [`union_array`]     | `awkward_UnionArray_*`                                               |
//! | [`reduce`]          | free-standing `awkward_reduce_*`                                     |
//! | [`sort`]            | `awkward_argsort`, `awkward_sort`, `awkward_sorting_ranges*`         |
//! | [`unique`]          | `awkward_unique_*`                                                   |
//! | [`misc`]            | `awkward_localindex`, `awkward_missing_repeat`                       |
//!
//! Conventions used by every kernel module:
//! * Slices instead of raw pointer + length pairs.
//! * `Result<(), KernelError>` instead of an opaque `ERROR` integer.
//! * Generic functions with trait bounds instead of hand-rolled C++
//!   template specialisations.
//! * `#[inline]` on the generic core; named specialisation wrappers kept
//!   for ABI compatibility with the original C symbols.
//!
//! See `CPU_KERNELS_LAYOUT.md` at the crate root for the full design.

// ── Backend-shared infrastructure (will move up to `kernels::*` later) ──────
pub mod error;
pub mod kernel_utils;
pub mod unicode;
pub mod utils;

// ── Per-layout kernel modules ───────────────────────────────────────────────
pub mod content;
pub mod index;
pub mod list_array;
pub mod list_offset;
pub mod mask;
pub mod masked_array;
pub mod numpy_array;
pub mod record_array;
pub mod regular_array;
pub mod union_array;

// ── Free-standing kernel modules ────────────────────────────────────────────
pub mod misc;
pub mod reduce;
pub mod sort;
pub mod unique;
