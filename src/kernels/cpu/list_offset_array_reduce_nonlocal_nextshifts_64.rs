// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build the shifts array for a non-local reduction over a ListOffsetArray.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64.cpp`.

/// Compute per-element shift values used for non-local reductions.
///
/// For each list `i`:
/// 1. If `starts[parents[i]] == i` (first list in its parent group), reset
///    `nummissing[k] = 0` for all `k`.
/// 2. For `k` in `count..maxcount`: `nummissing[k]++` (padding positions).
/// 3. For `j` in `start..stop`: `missing[start+j] = nummissing[j]`.
///
/// Finally, for each output position `j` in `0..nextlen`:
/// `nextshifts[j] = missing[nextcarry[j]]`.
///
/// # Parameters
///
/// * `nummissing`  – Scratch buffer of length `maxcount` (zeroed internally per group).
/// * `missing`     – Scratch flat buffer of length ≥ `offsets.last()`.
/// * `nextshifts`  – Output shifts; length `nextlen`.
/// * `offsets`     – List boundaries (length `length+1`).
/// * `starts`      – First list index per parent group.
/// * `parents`     – Parent-group index per list.
/// * `maxcount`    – Maximum list length.
/// * `nextcarry`   – The carry indices for the next reduction step.
pub fn list_offset_array_reduce_nonlocal_nextshifts_64(
    nummissing: &mut [i64],
    missing: &mut [i64],
    nextshifts: &mut [i64],
    offsets: &[i64],
    starts: &[i64],
    parents: &[i64],
    maxcount: usize,
    nextcarry: &[i64],
) {
    let length = parents.len();
    assert!(nummissing.len() >= maxcount);

    for i in 0..length {
        let start = offsets[i] as usize;
        let stop = offsets[i + 1] as usize;
        let count = stop - start;
        let parent = parents[i] as usize;

        // Reset nummissing at the start of each parent group.
        if starts[parent] as usize == i {
            for k in 0..maxcount {
                nummissing[k] = 0;
            }
        }

        // Accumulate padding counts.
        for k in count..maxcount {
            nummissing[k] += 1;
        }

        // Record the shift for each real element.
        for j in 0..count {
            missing[start + j] = nummissing[j];
        }
    }

    for j in 0..nextcarry.len() {
        nextshifts[j] = missing[nextcarry[j] as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_two_lists_same_parent() {
        // offsets=[0,2,3]; two lists of lengths 2 and 1; maxcount=2; parents=[0,0]; starts=[0]
        // List 0 (i=0): parent=0, starts[0]=0==i → reset nummissing=[0,0]
        //   count=2 < maxcount=2 → no padding increments
        //   missing[0]=0, missing[1]=0
        // List 1 (i=1): parent=0, starts[0]=0≠1 → don't reset
        //   count=1 < maxcount=2 → nummissing[1]++  → [0,1]
        //   missing[2]=0
        // nextcarry=[0,1,2], nextshifts should be [0,0,0]
        let offsets = [0i64, 2, 3];
        let starts = [0i64];
        let parents = [0i64, 0];
        let nextcarry = [0i64, 1, 2];
        let mut nummissing = [0i64; 2];
        let mut missing = [0i64; 3];
        let mut nextshifts = [0i64; 3];
        list_offset_array_reduce_nonlocal_nextshifts_64(
            &mut nummissing,
            &mut missing,
            &mut nextshifts,
            &offsets,
            &starts,
            &parents,
            2,
            &nextcarry,
        );
        assert_eq!(nextshifts, [0, 0, 0]);
    }

    #[test]
    fn padding_increments() {
        // Two lists: lengths 1 and 3; maxcount=3; same parent
        // List 0 (count=1): reset; nummissing=[0,0,0]; nummissing[1]++,nummissing[2]++ → [0,1,1]; missing[0]=0
        // List 1 (count=3): nummissing stays [0,1,1]; missing[1..4]=[0,1,1]
        let offsets = [0i64, 1, 4];
        let starts = [0i64];
        let parents = [0i64, 0];
        let nextcarry = [0i64, 1, 2, 3];
        let mut nummissing = [0i64; 3];
        let mut missing = [0i64; 4];
        let mut nextshifts = [0i64; 4];
        list_offset_array_reduce_nonlocal_nextshifts_64(
            &mut nummissing,
            &mut missing,
            &mut nextshifts,
            &offsets,
            &starts,
            &parents,
            3,
            &nextcarry,
        );
        assert_eq!(nextshifts, [0, 0, 1, 1]);
    }
}
