// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Adjust argmin/argmax results from global indices to within-list offsets.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_NumpyArray_reduce_adjust_starts_64.cpp`.

/// For each output group `k`:
/// * If `toptr[k] >= 0`, subtract `starts[parents[toptr[k]]]` so the result
///   is a within-list position rather than a global flat index.
///
/// This is called after `reduce_argmin` / `reduce_argmax` to normalise the
/// global indices they return into local ones.
pub fn numpy_array_reduce_adjust_starts_64(toptr: &mut [i64], parents: &[i64], starts: &[i64]) {
    for v in toptr.iter_mut() {
        let i = *v;
        if i >= 0 {
            let parent = parents[i as usize] as usize;
            let start = starts[parent];
            *v += -start;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_adjustment() {
        // Global argmin indices: [2, 5]; parents for those flat positions: [0, 1]
        // starts: [2, 4] → local = 2-2=0, 5-4=1
        let parents = [0i64, 0, 0, 1, 1, 1]; // flat array's parent map
        let starts = [0i64, 3];
        let mut toptr = [2i64, 5]; // argmin results (global indices)
        numpy_array_reduce_adjust_starts_64(&mut toptr, &parents, &starts);
        assert_eq!(toptr, [2, 2]); // 2-0=2, 5-3=2
    }

    #[test]
    fn minus_one_unchanged() {
        let parents = [0i64];
        let starts = [0i64];
        let mut toptr = [-1i64]; // empty group
        numpy_array_reduce_adjust_starts_64(&mut toptr, &parents, &starts);
        assert_eq!(toptr, [-1]);
    }
}
