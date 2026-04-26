// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! RegularArray kernels.

pub mod combinations;
pub mod getitem_carry;
pub mod getitem_jagged_expand;
pub mod getitem_next_array;
pub mod getitem_next_array_advanced;
pub mod getitem_next_array_regularize;
pub mod getitem_next_at;
pub mod getitem_next_range;
pub mod getitem_next_range_spreadadvanced;
pub mod localindex;
pub mod reduce_local_nextparents;
pub mod reduce_nonlocal_preparenext;
pub mod rpad_and_clip_axis1;
