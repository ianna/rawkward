// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! ListArray kernels.

pub mod broadcast_tooffsets;
pub mod combinations;
pub mod combinations_length;
pub mod compact_offsets;
pub mod fill;

pub mod getitem_jagged_apply;
pub mod getitem_jagged_carrylen;
pub mod getitem_jagged_descend;
pub mod getitem_jagged_expand;
pub mod getitem_jagged_numvalid;
pub mod getitem_jagged_shrink;
pub mod getitem_next_array;
pub mod getitem_next_array_advanced;
pub mod getitem_next_at;
pub mod getitem_next_range;
pub mod getitem_next_range_carrylength;
pub mod getitem_next_range_counts;
pub mod getitem_next_range_spreadadvanced;

pub mod localindex;
pub mod min_range;
pub mod rpad_and_clip_length_axis1;
pub mod rpad_axis1;
pub mod validity;
