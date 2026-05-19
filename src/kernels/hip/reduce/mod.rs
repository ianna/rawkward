// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Re-export HipDtype from sum so the other modules can import it with
// `use crate::kernels::hip::reduce::HipDtype;`
pub use sum::HipDtype;

// Scalar reducers — output has one element per segment
pub mod sum;
pub mod min;
pub mod max;
pub mod argmin;
pub mod argmax;
pub mod prod;
pub mod count;
pub mod countnonzero;
pub mod unique_offsets;

// Data-level operations — output has one element per input data position
pub mod sort;
pub mod sorting_ranges;
pub mod unique_ranges;
