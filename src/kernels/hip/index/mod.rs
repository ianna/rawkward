// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// -----------------------------------------------------------------------
// Index-level operations (IndexedArray / IndexedOptionArray)
// -----------------------------------------------------------------------

/// Replace -1 (None) entries with sequential indices beyond existing values.
pub mod index_nones_as_index;

/// Validate the index array of an IndexedArray or IndexedOptionArray.
pub mod indexed_validity;

/// Overlay a byte mask on an index array, turning masked positions to -1.
pub mod indexed_overlay_mask;

/// Fill a slice of an output index from a source index (with base offset).
/// Also: fill a slice with base, base+1, base+2, …
pub mod indexed_fill;

/// Count null (negative) entries; per-element null flags; unique-index fill.
pub mod indexed_numnull;

/// Compose two index arrays into a single flattened index.
pub mod indexed_simplify;

/// Collect carry indices for indexed getitem; split into carry and outindex.
pub mod indexed_getitem_nextcarry;

/// Flatten an IndexedOptionArray: collect non-null carry; build None→empty offsets.
pub mod indexed_flatten;

/// Collect within-list positions of null entries.
pub mod indexed_index_of_nulls;

/// Range-based carry collection and tostarts/tostops for IndexedArray ranges.
pub mod indexed_ranges;

/// Build carry/parents/outindex for reduction; fix output offsets;
/// compute non-local nextshifts (with and without incoming shifts).
pub mod indexed_reduce_next;

/// Build carry indices for the local-reduction preparation step.
pub mod indexed_local_preparenext;

/// Build output index and offsets for the unique-values reduction step.
pub mod indexed_unique;

/// rpad-and-clip index builders: mask-axis1, axis0, axis1.
pub mod indexedoption_rpad_and_clip_mask_axis1;
