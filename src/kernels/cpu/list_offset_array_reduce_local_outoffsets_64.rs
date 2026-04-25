// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build output offsets from a parents array for a local reduction.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_reduce_local_outoffsets_64.cpp`.

/// Construct `outoffsets` from a `parents` array.
///
/// For each position `i` in `parents`, emit the current `i` into `outoffsets`
/// for every group that `parents[i]` skips over since the last seen parent.
/// After exhausting `parents`, fill the remaining `outoffsets` slots with
/// `lenparents`.
///
/// # Parameters
///
/// * `outoffsets`  – Output; length must be `outlength + 1`.
/// * `parents`     – Per-element group index (non-decreasing).
/// * `outlength`   – Number of output groups.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_offset_array_reduce_local_outoffsets_64::list_offset_array_reduce_local_outoffsets_64;
///
/// let parents = [0i64, 0, 1, 1, 1, 2];
/// let mut offsets = [0i64; 4];
/// list_offset_array_reduce_local_outoffsets_64(&mut offsets, &parents, 3);
/// assert_eq!(offsets, [0, 2, 5, 6]);
/// ```
pub fn list_offset_array_reduce_local_outoffsets_64(
    outoffsets: &mut [i64],
    parents: &[i64],
    outlength: usize,
) {
    assert!(outoffsets.len() >= outlength + 1);
    let lenparents = parents.len() as i64;
    let mut k = 0i64;
    let mut last = -1i64;

    for (i, &p) in parents.iter().enumerate() {
        while last < p {
            outoffsets[k as usize] = i as i64;
            k += 1;
            last += 1;
        }
    }
    while (k as usize) <= outlength {
        outoffsets[k as usize] = lenparents;
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let parents = [0i64, 0, 1, 1, 1, 2];
        let mut offsets = [0i64; 4];
        list_offset_array_reduce_local_outoffsets_64(&mut offsets, &parents, 3);
        assert_eq!(offsets, [0, 2, 5, 6]);
    }

    #[test]
    fn empty_last_group() {
        // Parent 2 never appears — its offset range is [5,5] (empty)
        let parents = [0i64, 0, 1, 1];
        let mut offsets = [0i64; 4];
        list_offset_array_reduce_local_outoffsets_64(&mut offsets, &parents, 3);
        assert_eq!(offsets, [0, 2, 4, 4]);
    }

    #[test]
    fn single_group() {
        let parents = [0i64, 0, 0];
        let mut offsets = [0i64; 2];
        list_offset_array_reduce_local_outoffsets_64(&mut offsets, &parents, 1);
        assert_eq!(offsets, [0, 3]);
    }

    #[test]
    fn all_different_groups() {
        let parents = [0i64, 1, 2];
        let mut offsets = [0i64; 4];
        list_offset_array_reduce_local_outoffsets_64(&mut offsets, &parents, 3);
        assert_eq!(offsets, [0, 1, 2, 3]);
    }
}
