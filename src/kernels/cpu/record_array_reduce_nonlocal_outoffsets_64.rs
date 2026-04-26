// Copyright (c) 2026 Ianna Osborne 
// SPDX-License-Identifier: BSD-3-Clause

//! Build output offsets and carry for a non-local reduction over a RecordArray.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_RecordArray_reduce_nonlocal_outoffsets_64.cpp`.

/// Populate `outoffsets[0..=outlength]` and `outcarry[0..outlength]`.
///
/// Scans `parents` for contiguous runs and writes the stop index of each run
/// into `outoffsets`.  `outcarry[parents[run_start]]` records which sublist
/// index corresponds to that parent group.  Missing parents get carry values
/// pointing to appended empty lists.
pub fn record_array_reduce_nonlocal_outoffsets_64(
    outoffsets: &mut [i64],
    outcarry: &mut [i64],
    parents: &[i64],
    outlength: usize,
) {
    assert!(outoffsets.len() >= outlength + 1);
    assert!(outcarry.len()  >= outlength + 1);

    let lenparents = parents.len() as i64;

    outoffsets[0] = 0;
    for v in outcarry.iter_mut() { *v = -1; }

    let mut i = 0usize;
    let mut j_stop = 1usize;
    let mut k_sublist = 0usize;

    while j_stop < parents.len() {
        if parents[i] != parents[j_stop] {
            outoffsets[k_sublist + 1] = j_stop as i64;
            outcarry[parents[i] as usize] = k_sublist as i64;
            i = j_stop;
            k_sublist += 1;
        }
        j_stop += 1;
    }
    // Close the last sublist.
    if !parents.is_empty() {
        outoffsets[k_sublist + 1] = j_stop as i64;
        outcarry[parents[i] as usize] = k_sublist as i64;
        k_sublist += 1;
    }

    // Append empty lists for missing parents.
    for idx in k_sublist..outlength {
        outoffsets[idx + 1] = lenparents;
    }

    // Replace sentinel (-1) with index of appended empty lists.
    let mut next_empty = k_sublist as i64;
    for idx in 0..=outlength {
        if outcarry[idx] == -1 {
            outcarry[idx] = next_empty;
            next_empty += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_groups() {
        // parents=[0,0,1,1], outlength=2
        let parents = [0i64, 0, 1, 1];
        let mut offsets = [0i64; 3];
        let mut carry   = [0i64; 3];
        record_array_reduce_nonlocal_outoffsets_64(&mut offsets, &mut carry, &parents, 2);
        assert_eq!(offsets, [0, 2, 4]);
        assert_eq!(&carry[..2], &[0, 1]);
    }

    #[test]
    fn missing_parent_gets_empty_list() {
        // parents=[0,0], outlength=3 → parent 1 and 2 are missing
        let parents = [0i64, 0];
        let mut offsets = [0i64; 4];
        let mut carry   = [0i64; 4];
        record_array_reduce_nonlocal_outoffsets_64(&mut offsets, &mut carry, &parents, 3);
        assert_eq!(offsets[1], 2);
        // Missing parents get carry ≥ 1
        assert!(carry[1] >= 1);
        assert!(carry[2] >= 1);
    }
}
