//! # cpu-kernels
//!
//! Pure-Rust implementations of the Awkward Array CPU kernels originally
//! written in C++.  Each module corresponds to one source file from the
//! original `src/cpu-kernels/` directory and exposes the same public API
//! (function names, argument order, semantics) while following idiomatic
//! Rust conventions:
//!
//! * Slices instead of raw pointer + length pairs.
//! * `Result<(), KernelError>` instead of an opaque `ERROR` integer.
//! * Generic functions with trait bounds instead of hand-rolled template
//!   specialisations.
//! * `#[inline]` on the generic core; named specialisation wrappers kept for
//!   ABI compatibility where desired.

// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

pub mod index;

pub mod error;
pub mod kernel_utils;
pub mod unicode;
pub mod utils;

// ── Mask / option array kernels ──────────────────────────────────────────────
pub mod bit_masked_array_to_byte_masked_array;
pub mod bit_masked_array_to_indexed_option_array;
pub mod byte_masked_array_getitem_nextcarry;
pub mod byte_masked_array_getitem_nextcarry_outindex;
pub mod byte_masked_array_numnull;
pub mod byte_masked_array_overlay_mask;
pub mod byte_masked_array_reduce_next_64;
pub mod byte_masked_array_reduce_next_nonlocal_nextshifts_64;
pub mod byte_masked_array_reduce_next_nonlocal_nextshifts_fromshifts_64;
pub mod byte_masked_array_to_indexed_option_array;

// ── Content / missing-jagged ─────────────────────────────────────────────────
pub mod content_getitem_next_missing_jagged_getmaskstartstop;

// ── Index kernels ────────────────────────────────────────────────────────────
pub mod index_nones_as_index;

// ── IndexedArray kernels ─────────────────────────────────────────────────────
pub mod indexed_array_fill;
pub mod indexed_array_fill_count;
pub mod indexed_array_flatten_nextcarry;
pub mod indexed_array_flatten_none2empty;
pub mod indexed_array_getitem_nextcarry;
pub mod indexed_array_getitem_nextcarry_outindex;
pub mod indexed_array_index_of_nulls;
pub mod indexed_array_local_preparenext_64;
pub mod indexed_array_numnull;
pub mod indexed_array_numnull_parents;
pub mod indexed_array_numnull_unique_64;
pub mod indexed_array_overlay_mask;
pub mod indexed_array_ranges_carry_next_64;
pub mod indexed_array_ranges_next_64;
pub mod indexed_array_reduce_next_64;
pub mod indexed_array_reduce_next_fix_offsets_64;
pub mod indexed_array_reduce_next_nonlocal_nextshifts_64;
pub mod indexed_array_reduce_next_nonlocal_nextshifts_fromshifts_64;
pub mod indexed_array_simplify;
pub mod indexed_array_unique_next_index_and_offsets_64;
pub mod indexed_array_validity;
pub mod indexed_option_array_rpad_and_clip_mask_axis1;

// ── ListArray kernels ─────────────────────────────────────────────────────────
pub mod list_array_broadcast_tooffsets;
pub mod list_array_combinations;
pub mod list_array_combinations_length;
pub mod list_array_compact_offsets;
pub mod list_array_fill;
pub mod list_array_getitem_jagged_apply;
pub mod list_array_getitem_jagged_carrylen;
pub mod list_array_getitem_jagged_descend;
pub mod list_array_getitem_jagged_expand;
pub mod list_array_getitem_jagged_numvalid;
pub mod list_array_getitem_jagged_shrink;
pub mod list_array_getitem_next_array;
pub mod list_array_getitem_next_array_advanced;
pub mod list_array_getitem_next_at;
pub mod list_array_getitem_next_range;
pub mod list_array_getitem_next_range_carrylength;
pub mod list_array_getitem_next_range_counts;
pub mod list_array_getitem_next_range_spreadadvanced;
pub mod list_array_localindex;
pub mod list_array_min_range;
pub mod list_array_rpad_and_clip_length_axis1;
pub mod list_array_rpad_axis1;
pub mod list_array_validity;

// ── ListOffsetArray kernels ───────────────────────────────────────────────────
pub mod list_offset_array_argsort_strings;
pub mod list_offset_array_drop_none_indexes;
pub mod list_offset_array_flatten_offsets;
pub mod list_offset_array_local_preparenext_64;
pub mod list_offset_array_reduce_local_nextparents_64;
pub mod list_offset_array_reduce_local_outoffsets_64;
pub mod list_offset_array_reduce_nonlocal_maxcount_offsetscopy_64;
pub mod list_offset_array_reduce_nonlocal_nextshifts_64;
pub mod list_offset_array_reduce_nonlocal_nextstarts_64;
pub mod list_offset_array_reduce_nonlocal_outstartsstops_64;
pub mod list_offset_array_reduce_nonlocal_preparenext_64;
pub mod list_offset_array_rpad_and_clip_axis1;
pub mod list_offset_array_rpad_axis1;
pub mod list_offset_array_rpad_length_axis1;
pub mod list_offset_array_to_regular_array;

// ── Sort / Argsort / Unique ───────────────────────────────────────────────────
pub mod argsort;
pub mod sort;
pub mod sorting_ranges;
pub mod sorting_ranges_length;
pub mod unique_offsets;
pub mod unique_ranges;
pub mod unique_ranges_bool;

// ── MaskedArray / RecordArray ─────────────────────────────────────────────────
pub mod masked_array_getitem_next_jagged_project;
pub mod record_array_reduce_nonlocal_outoffsets_64;

// ── NumpyArray string/UTF kernels ─────────────────────────────────────────────
pub mod numpy_array_prepare_utf8_to_utf32_padded;
pub mod numpy_array_sort_asstrings_uint8;
pub mod numpy_array_subrange_equal;
pub mod numpy_array_unique_strings_uint8;
pub mod numpy_array_utf8_to_utf32_padded;

// ── Complex reductions (all in one module) ────────────────────────────────────
pub mod reduce_complex;

// ── RegularArray kernels ──────────────────────────────────────────────────────
pub mod regular_array_combinations_64;
pub mod regular_array_getitem_carry;
pub mod regular_array_getitem_jagged_expand;
pub mod regular_array_getitem_next_array_advanced;
pub mod regular_array_getitem_next_array_regularize;
pub mod regular_array_getitem_next_array;
pub mod regular_array_getitem_next_at;
pub mod regular_array_getitem_next_range_spreadadvanced;
pub mod regular_array_getitem_next_range;
pub mod regular_array_localindex;
pub mod regular_array_reduce_local_nextparents_64;
pub mod regular_array_reduce_nonlocal_preparenext_64;
pub mod regular_array_rpad_and_clip_axis1;

// ── UnionArray kernels ────────────────────────────────────────────────────────
pub mod union_array_fillindex;
pub mod union_array_fillindex_count;
pub mod union_array_fillna;
pub mod union_array_filltags;
pub mod union_array_filltags_const;
pub mod union_array_flatten_combine;
pub mod union_array_flatten_length;
pub mod union_array_nestedfill_tags_index;
pub mod union_array_project;
pub mod union_array_regular_index;
pub mod union_array_regular_index_getsize;
pub mod union_array_simplify;
pub mod union_array_simplify_one;
pub mod union_array_validity;

// ── Reduction kernels ────────────────────────────────────────────────────────
pub mod localindex;
pub mod reduce_argmax;
pub mod reduce_argmin;
pub mod reduce_count_64;
pub mod reduce_countnonzero;
pub mod reduce_max;
pub mod reduce_min;
pub mod reduce_prod;
pub mod reduce_prod_bool;
pub mod reduce_sum;
pub mod reduce_sum_bool;
pub mod reduce_sum_bool_variants;

// ── NumpyArray kernels ────────────────────────────────────────────────────────
pub mod numpy_array_pad_zero_to_length;
pub mod numpy_array_rearrange_shifted;
pub mod numpy_array_reduce_adjust_starts_64;
pub mod numpy_array_reduce_adjust_starts_shifts_64;
pub mod numpy_array_reduce_mask_byte_masked_array_64;

// ── Index rpad/clip kernels ───────────────────────────────────────────────────
pub mod index_rpad_and_clip_axis0;
pub mod index_rpad_and_clip_axis1;
pub mod missing_repeat;
