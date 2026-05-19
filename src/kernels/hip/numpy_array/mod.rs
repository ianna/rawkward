// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// -----------------------------------------------------------------------
// Scalar / index-level operations (output same length as segments or data)
// -----------------------------------------------------------------------

/// Pad sublists to a fixed length with zeros.
pub mod pad_zero_to_length;

/// Adjust argmin/argmax global indices to within-list local positions.
pub mod reduce_adjust_starts;

/// Build a ByteMaskedArray validity mask from parent indices.
pub mod reduce_mask_bytemasked;

/// Two-pass rearrangement of a shifted index array.
pub mod rearrange_shifted;

// -----------------------------------------------------------------------
// UTF-8 / Unicode operations
// -----------------------------------------------------------------------

/// Compute the maximum codepoint count across all UTF-8 sublists.
pub mod prepare_utf8_to_utf32_padded;

/// Decode UTF-8 sublists into padded UTF-32 codepoint arrays.
pub mod utf8_to_utf32_padded;

// -----------------------------------------------------------------------
// String-as-bytes operations
// -----------------------------------------------------------------------

/// Check pairwise equality of same-length sub-ranges.
pub mod subrange_equal;

/// Sort variable-length strings (uint8) by lexicographic byte order.
pub mod sort_asstrings_uint8;

/// Deduplicate consecutive equal strings in a flat byte buffer.
pub mod unique_strings_uint8;
