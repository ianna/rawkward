// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Index, IndexedArray and IndexedOptionArray kernels.

// Free-standing index helpers.
pub mod nones_as_index;
pub mod rpad_and_clip_axis0;
pub mod rpad_and_clip_axis1;

// IndexedArray kernels.
pub mod indexed_fill;
pub mod indexed_fill_count;
pub mod indexed_flatten_nextcarry;
pub mod indexed_flatten_none2empty;
pub mod indexed_getitem_nextcarry;
pub mod indexed_getitem_nextcarry_outindex;
pub mod indexed_index_of_nulls;
pub mod indexed_local_preparenext;
pub mod indexed_numnull;
pub mod indexed_numnull_parents;
pub mod indexed_numnull_unique;
pub mod indexed_overlay_mask;
pub mod indexed_ranges_carry_next;
pub mod indexed_ranges_next;
pub mod indexed_reduce_next;
pub mod indexed_reduce_next_fix_offsets;
pub mod indexed_reduce_next_nonlocal_nextshifts;
pub mod indexed_reduce_next_nonlocal_nextshifts_fromshifts;
pub mod indexed_simplify;
pub mod indexed_unique_next_index_and_offsets;
pub mod indexed_validity;

// IndexedOptionArray kernels.
pub mod indexedoption_rpad_and_clip_mask_axis1;
