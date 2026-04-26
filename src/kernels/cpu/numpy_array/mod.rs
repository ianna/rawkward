// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! NumpyArray kernels.

pub mod pad_zero_to_length;
pub mod prepare_utf8_to_utf32_padded;
pub mod rearrange_shifted;
pub mod reduce_adjust_starts;
pub mod reduce_adjust_starts_shifts;
pub mod reduce_mask_bytemasked;
pub mod sort_asstrings_uint8;
pub mod subrange_equal;
pub mod unique_strings_uint8;
pub mod utf8_to_utf32_padded;
