// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Fix the output offsets array after a reduction step over an IndexedArray.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_IndexedArray_reduce_next_fix_offsets_64.cpp`.

/// Copy `starts[0..startslength]` into `outoffsets[0..startslength]` and
/// write `outindexlength` as the final sentinel `outoffsets[startslength]`.
///
/// This kernel converts a `starts` array (which holds the beginning of each
/// output group) into a complete offsets array by appending the total output
/// length as the closing bound.
///
/// # Parameters
///
/// * `outoffsets`     – Output slice of length `startslength + 1`.
/// * `starts`         – Per-group start positions; length `startslength`.
/// * `outindexlength` – Total length of the output index (written as the
///                      final entry).
///
/// # Panics
///
/// Panics if `outoffsets.len() < startslength + 1` or `starts.len() < startslength`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_reduce_next_fix_offsets_64::indexed_array_reduce_next_fix_offsets_64;
///
/// let starts = [0i64, 3, 5];
/// let mut offsets = [0i64; 4];
/// indexed_array_reduce_next_fix_offsets_64(&mut offsets, &starts, 8);
/// assert_eq!(offsets, [0, 3, 5, 8]);
/// ```
pub fn indexed_array_reduce_next_fix_offsets_64(
    outoffsets: &mut [i64],
    starts: &[i64],
    outindexlength: i64,
) {
    let startslength = starts.len();
    assert!(
        outoffsets.len() >= startslength + 1,
        "outoffsets must hold startslength+1 elements"
    );
    outoffsets[..startslength].copy_from_slice(starts);
    outoffsets[startslength] = outindexlength;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let starts = [0i64, 3, 5];
        let mut offsets = [0i64; 4];
        indexed_array_reduce_next_fix_offsets_64(&mut offsets, &starts, 8);
        assert_eq!(offsets, [0, 3, 5, 8]);
    }

    #[test]
    fn single_group() {
        let starts = [0i64];
        let mut offsets = [0i64; 2];
        indexed_array_reduce_next_fix_offsets_64(&mut offsets, &starts, 5);
        assert_eq!(offsets, [0, 5]);
    }

    #[test]
    fn zero_groups() {
        let starts: [i64; 0] = [];
        let mut offsets = [0i64; 1];
        indexed_array_reduce_next_fix_offsets_64(&mut offsets, &starts, 42);
        assert_eq!(offsets[0], 42);
    }

    #[test]
    fn copies_starts_verbatim() {
        let starts = [10i64, 20, 30];
        let mut offsets = [0i64; 4];
        indexed_array_reduce_next_fix_offsets_64(&mut offsets, &starts, 35);
        assert_eq!(offsets, [10, 20, 30, 35]);
    }
}
