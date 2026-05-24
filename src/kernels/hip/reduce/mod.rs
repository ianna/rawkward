// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// `HipDtype` is the shared dtype tag used by all reduce modules.  It lives in
// `argmin` (its first consumer) and is re-exported here so every sibling can
// import it as `crate::kernels::hip::reduce::HipDtype`.
pub use argmin::HipDtype;

// Scalar reducers — output has one element per segment
pub mod argmax;
pub mod argmin;
pub mod count;
pub mod countnonzero;
pub mod max;
pub mod min;
pub mod prod;
pub mod sum;
pub mod unique_offsets;

// Data-level operations — output has one element per input data position
pub mod sort;
pub mod sorting_ranges;
pub mod unique_ranges;
