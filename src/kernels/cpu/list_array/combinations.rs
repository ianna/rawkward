//! Generate all n-combinations of indices for every list in a ListArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_combinations.cpp`.

// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::kernels::cpu::kernel_utils::list_array_combinations_step_64;

/// For each list `i` spanning `starts[i]..stops[i]`, enumerate all `n`-element
/// combinations (with or without replacement) and write the `n` column arrays
/// into `tocarry`.
///
/// `tocarry` must be a slice of `n` separate `Vec<i64>` buffers, each
/// pre-allocated to hold the total number of combinations across all lists
/// (use [`crate::list_array_combinations_length`] to compute this).  
/// `toindex` and `fromindex` are working buffers of length `n`.
///
/// # Parameters
///
/// * `tocarry`     – `n` output buffers; `tocarry[k]` receives the k-th element
///                   of each combination.
/// * `toindex`     – Mutable scratch buffer of length `n` (write positions per column).
/// * `fromindex`   – Mutable scratch buffer of length `n` (current combination state).
/// * `n`           – Number of elements per combination.
/// * `replacement` – Whether combinations with repeated elements are allowed.
/// * `starts`/`stops` – List boundaries.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_array_combinations::list_array64_combinations_64;
///
/// // Single list [0,1,2]; 2-combinations → (0,1),(0,2),(1,2)
/// let starts = [0i64];
/// let stops  = [3i64];
/// let mut col0 = vec![0i64; 3];
/// let mut col1 = vec![0i64; 3];
/// let mut tocarry   = vec![col0, col1];
/// let mut toindex   = vec![0usize; 2];
/// let mut fromindex = vec![0i64;  2];
/// list_array64_combinations_64(&mut tocarry, &mut toindex, &mut fromindex, 2, false, &starts, &stops);
/// assert_eq!(tocarry[0], vec![0, 0, 1]);
/// assert_eq!(tocarry[1], vec![1, 2, 2]);
/// ```
pub fn list_array_combinations<C>(
    tocarry: &mut Vec<Vec<i64>>,
    toindex: &mut Vec<usize>,
    fromindex: &mut Vec<i64>,
    n: usize,
    replacement: bool,
    starts: &[C],
    stops: &[C],
) where
    C: Copy + Into<i64>,
{
    assert_eq!(starts.len(), stops.len());
    assert_eq!(tocarry.len(), n);
    assert_eq!(toindex.len(), n);
    assert_eq!(fromindex.len(), n);

    // Reset write positions.
    for v in toindex.iter_mut() {
        *v = 0;
    }

    for i in 0..starts.len() {
        let start: i64 = starts[i].into();
        let stop: i64 = stops[i].into();
        fromindex[0] = start;
        list_array_combinations_step_64(tocarry, toindex, fromindex, 0, stop, n, replacement);
    }
}

pub fn list_array32_combinations_64(
    tocarry: &mut Vec<Vec<i64>>,
    toindex: &mut Vec<usize>,
    fromindex: &mut Vec<i64>,
    n: usize,
    replacement: bool,
    starts: &[i32],
    stops: &[i32],
) {
    list_array_combinations(tocarry, toindex, fromindex, n, replacement, starts, stops);
}
pub fn list_array_u32_combinations_64(
    tocarry: &mut Vec<Vec<i64>>,
    toindex: &mut Vec<usize>,
    fromindex: &mut Vec<i64>,
    n: usize,
    replacement: bool,
    starts: &[u32],
    stops: &[u32],
) {
    list_array_combinations(tocarry, toindex, fromindex, n, replacement, starts, stops);
}
pub fn list_array64_combinations_64(
    tocarry: &mut Vec<Vec<i64>>,
    toindex: &mut Vec<usize>,
    fromindex: &mut Vec<i64>,
    n: usize,
    replacement: bool,
    starts: &[i64],
    stops: &[i64],
) {
    list_array_combinations(tocarry, toindex, fromindex, n, replacement, starts, stops);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_single_list() {
        // [0,1,2] → (0,1),(0,2),(1,2)
        let starts = [0i64];
        let stops = [3i64];
        let mut tocarry = vec![vec![0i64; 3], vec![0i64; 3]];
        let mut toindex = vec![0usize; 2];
        let mut fromindex = vec![0i64; 2];
        list_array64_combinations_64(
            &mut tocarry,
            &mut toindex,
            &mut fromindex,
            2,
            false,
            &starts,
            &stops,
        );
        assert_eq!(tocarry[0], vec![0, 0, 1]);
        assert_eq!(tocarry[1], vec![1, 2, 2]);
    }

    #[test]
    fn pairs_with_replacement() {
        // [0,1,2] with replacement → (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
        let starts = [0i64];
        let stops = [3i64];
        let mut tocarry = vec![vec![0i64; 6], vec![0i64; 6]];
        let mut toindex = vec![0usize; 2];
        let mut fromindex = vec![0i64; 2];
        list_array64_combinations_64(
            &mut tocarry,
            &mut toindex,
            &mut fromindex,
            2,
            true,
            &starts,
            &stops,
        );
        assert_eq!(tocarry[0], vec![0, 0, 0, 1, 1, 2]);
        assert_eq!(tocarry[1], vec![0, 1, 2, 1, 2, 2]);
    }

    #[test]
    fn two_lists() {
        // [0,1] → (0,1); [2,3] → (2,3); total 2 pairs
        let starts = [0i64, 2];
        let stops = [2i64, 4];
        let mut tocarry = vec![vec![0i64; 2], vec![0i64; 2]];
        let mut toindex = vec![0usize; 2];
        let mut fromindex = vec![0i64; 2];
        list_array64_combinations_64(
            &mut tocarry,
            &mut toindex,
            &mut fromindex,
            2,
            false,
            &starts,
            &stops,
        );
        assert_eq!(tocarry[0], vec![0, 2]);
        assert_eq!(tocarry[1], vec![1, 3]);
    }

    #[test]
    fn triples() {
        // [0,1,2,3] → C(4,3)=4 triples
        let starts = [0i64];
        let stops = [4i64];
        let mut tocarry = vec![vec![0i64; 4], vec![0i64; 4], vec![0i64; 4]];
        let mut toindex = vec![0usize; 3];
        let mut fromindex = vec![0i64; 3];
        list_array64_combinations_64(
            &mut tocarry,
            &mut toindex,
            &mut fromindex,
            3,
            false,
            &starts,
            &stops,
        );
        assert_eq!(tocarry[0], vec![0, 0, 0, 1]);
        assert_eq!(tocarry[1], vec![1, 1, 2, 2]);
        assert_eq!(tocarry[2], vec![2, 3, 3, 3]);
    }
}
