//! Build the output index and offsets for the next unique-values step of an IndexedArray.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_IndexedArray_unique_next_index_and_offsets_64.cpp`.

// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

/// Fill `toindex` and `tooffsets` for the next unique-values reduction step.
///
/// The algorithm scans `startslength` groups defined by consecutive pairs in
/// `fromoffsets`.  For each element `j` inside group `i`:
/// * `toindex[k] = ll` (sequential output position), `k++`, `ll++`.
///
/// After processing each group, if `fromnulls[k] == 1` (the slot just past the
/// last written element of that group is marked null), a null sentinel is
/// inserted:
/// * `toindex[k] = -1`, `k++`, `shift++`.
///
/// `tooffsets[i+1] = fromoffsets[i+1] + shift` tracks how the group boundaries
/// have been extended by the inserted nulls.
///
/// # Parameters
///
/// * `toindex`      – Output index array; must be large enough for all valid
///                    entries plus one null per group that needs one.
/// * `tooffsets`    – Output offsets; length `startslength + 1`.
/// * `fromoffsets`  – Input group boundaries; length `startslength + 1`.
/// * `fromnulls`    – Per-position null flags (1 = null).  Indexed by the
///                    running `k` counter, i.e. it runs over the *output*
///                    positions.
///
/// # Panics
///
/// Panics if any slice access goes out of bounds.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_unique_next_index_and_offsets_64::indexed_array_unique_next_index_and_offsets_64;
///
/// // One group of 2 elements; fromnulls[2]=0 → no null appended.
/// let fromoffsets = [0i64, 2];
/// let fromnulls   = [0i64, 0, 0]; // checked at k=2 after the group
/// let mut toindex   = [0i64; 3];
/// let mut tooffsets = [0i64; 2];
/// indexed_array_unique_next_index_and_offsets_64(
///     &mut toindex, &mut tooffsets, &fromoffsets, &fromnulls, 1,
/// );
/// assert_eq!(&toindex[..2], &[0, 1]);
/// assert_eq!(tooffsets, [0, 2]);
/// ```
pub fn indexed_array_unique_next_index_and_offsets_64(
    toindex: &mut [i64],
    tooffsets: &mut [i64],
    fromoffsets: &[i64],
    fromnulls: &[i64],
    startslength: usize,
) {
    assert!(tooffsets.len() >= startslength + 1);

    let mut k = 0usize;
    let mut ll = 0i64;
    let mut shift = 0i64;

    toindex[0] = ll;
    tooffsets[0] = fromoffsets[0];

    for i in 0..startslength {
        let seg_start = fromoffsets[i] as usize;
        let seg_stop = fromoffsets[i + 1] as usize;
        for _j in seg_start..seg_stop {
            toindex[k] = ll;
            k += 1;
            ll += 1;
        }
        if fromnulls[k] == 1 {
            toindex[k] = -1;
            k += 1;
            shift += 1;
        }
        tooffsets[i + 1] = fromoffsets[i + 1] + shift;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_nulls() {
        // One group of 2 elements; fromnulls[2]=0 → no null inserted.
        let fromoffsets = [0i64, 2];
        let fromnulls = [0i64, 0, 0];
        let mut toindex = [0i64; 3];
        let mut tooffsets = [0i64; 2];
        indexed_array_unique_next_index_and_offsets_64(
            &mut toindex,
            &mut tooffsets,
            &fromoffsets,
            &fromnulls,
            1,
        );
        assert_eq!(&toindex[..2], &[0, 1]);
        assert_eq!(tooffsets, [0, 2]);
    }

    #[test]
    fn null_appended_after_group() {
        // One group of 2; fromnulls[2]=1 → null inserted after group.
        let fromoffsets = [0i64, 2];
        let fromnulls = [0i64, 0, 1]; // k=2 is null
        let mut toindex = [0i64; 4];
        let mut tooffsets = [0i64; 2];
        indexed_array_unique_next_index_and_offsets_64(
            &mut toindex,
            &mut tooffsets,
            &fromoffsets,
            &fromnulls,
            1,
        );
        assert_eq!(&toindex[..3], &[0, 1, -1]);
        // shift=1, so tooffsets[1] = fromoffsets[1] + 1 = 3
        assert_eq!(tooffsets, [0, 3]);
    }

    #[test]
    fn two_groups_no_nulls() {
        let fromoffsets = [0i64, 2, 4];
        let fromnulls = [0i64, 0, 0, 0, 0];
        let mut toindex = [0i64; 5];
        let mut tooffsets = [0i64; 3];
        indexed_array_unique_next_index_and_offsets_64(
            &mut toindex,
            &mut tooffsets,
            &fromoffsets,
            &fromnulls,
            2,
        );
        assert_eq!(&toindex[..4], &[0, 1, 2, 3]);
        assert_eq!(tooffsets, [0, 2, 4]);
    }

    #[test]
    fn two_groups_second_has_null() {
        // Group 0: [0,1] → k=2, fromnulls[2]=0 → no null.
        // Group 1: [2,3] → k=4, fromnulls[4]=1 → null.
        let fromoffsets = [0i64, 2, 4];
        let fromnulls = [0i64, 0, 0, 0, 1];
        let mut toindex = [0i64; 6];
        let mut tooffsets = [0i64; 3];
        indexed_array_unique_next_index_and_offsets_64(
            &mut toindex,
            &mut tooffsets,
            &fromoffsets,
            &fromnulls,
            2,
        );
        assert_eq!(&toindex[..5], &[0, 1, 2, 3, -1]);
        assert_eq!(tooffsets, [0, 2, 5]); // second group extended by 1
    }
}
