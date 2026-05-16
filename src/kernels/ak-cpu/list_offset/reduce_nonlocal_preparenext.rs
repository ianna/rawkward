// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Prepare the carry, parents, and distincts arrays for a non-local reduction.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_preparenext_64.cpp`.

/// Build `nextcarry`, `nextparents`, and `distincts` for a non-local
/// (strided/padded) reduction pass.
///
/// The algorithm interleaves `maxcount` round-robin passes over all `length`
/// lists: on each pass `j`, it picks the next available element from each
/// list (tracked by `offsetscopy`) and assigns it a globally-unique parent
/// slot `parent * maxcount + diff`.
///
/// # Parameters
///
/// * `nextcarry`        – Output flat content positions; length `nextlen`.
/// * `nextparents`      – Output parent slots; length `nextlen`.
/// * `maxnextparents`   – Output: maximum parent slot written.
/// * `distincts`        – Output mapping from parent-slot → column; length
///                        `distinctslen`.  Initialised to `-1` on entry.
/// * `offsetscopy`      – Mutable copy of `offsets[0..length]`; advanced in-place.
/// * `offsets`          – Original list boundaries (length `length+1`).
/// * `parents`          – Per-list parent-group index.
/// * `maxcount`         – Maximum list length.
///
/// # Returns
///
/// The value of `maxnextparents`.
pub fn list_offset_array_reduce_nonlocal_preparenext_64(
    nextcarry: &mut [i64],
    nextparents: &mut [i64],
    distincts: &mut [i64],
    offsetscopy: &mut [i64],
    offsets: &[i64],
    parents: &[i64],
    nextlen: usize,
    maxcount: i64,
) -> i64 {
    let length = parents.len();
    assert!(offsets.len() > length);
    assert!(offsetscopy.len() >= length);

    for d in distincts.iter_mut() {
        *d = -1;
    }

    let mut k = 0usize;
    let mut maxnextparents = 0i64;

    while k < nextlen {
        let mut j = 0i64;
        for i in 0..length {
            if offsetscopy[i] < offsets[i + 1] {
                let diff = offsetscopy[i] - offsets[i];
                let parent = parents[i];

                nextcarry[k] = offsetscopy[i];
                nextparents[k] = parent * maxcount + diff;

                if nextparents[k] > maxnextparents {
                    maxnextparents = nextparents[k];
                }

                let slot = nextparents[k] as usize;
                if distincts[slot] == -1 {
                    distincts[slot] = j;
                    j += 1;
                }

                k += 1;
                offsetscopy[i] += 1;
            }
        }
    }
    maxnextparents
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_lists_same_parent() {
        // offsets=[0,2,4], parents=[0,0], maxcount=2
        // nextlen=4 (2 lists × 2 elements each)
        let offsets = [0i64, 2, 4];
        let parents = [0i64, 0];
        let mut offsetscopy = [0i64, 2];
        let mut nextcarry = [0i64; 4];
        let mut nextparents = [0i64; 4];
        let mut distincts = [-1i64; 4]; // maxcount*nparents=2*2=4
        let max = list_offset_array_reduce_nonlocal_preparenext_64(
            &mut nextcarry,
            &mut nextparents,
            &mut distincts,
            &mut offsetscopy,
            &offsets,
            &parents,
            4,
            2,
        );
        // Round 0: list0→carry=0, parent=0*2+0=0; list1→carry=2, parent=0*2+0=0 (same slot!)
        // Round 1: list0→carry=1, parent=0*2+1=1; list1→carry=3, parent=0*2+1=1
        assert_eq!(&nextcarry, &[0, 2, 1, 3]);
        assert_eq!(&nextparents, &[0, 0, 1, 1]);
        assert!(max >= 1);
    }

    #[test]
    fn single_list() {
        let offsets = [0i64, 3];
        let parents = [0i64];
        let mut offsetscopy = [0i64];
        let mut nextcarry = [0i64; 3];
        let mut nextparents = [0i64; 3];
        let mut distincts = [-1i64; 3];
        list_offset_array_reduce_nonlocal_preparenext_64(
            &mut nextcarry,
            &mut nextparents,
            &mut distincts,
            &mut offsetscopy,
            &offsets,
            &parents,
            3,
            3,
        );
        assert_eq!(&nextcarry, &[0, 1, 2]);
        assert_eq!(&nextparents, &[0, 1, 2]);
    }
}
