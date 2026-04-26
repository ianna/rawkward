// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Generate all n-combinations of indices for a RegularArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_RegularArray_combinations_64.cpp`.

use crate::kernels::cpu::kernel_utils::list_array_combinations_step_64;

/// For each row `i` of a RegularArray (rows of fixed `size`), enumerate all
/// `n`-element combinations (with or without replacement) and write the `n`
/// column arrays into `tocarry`.
///
/// Each row spans content positions `i*size .. i*size + size`, so `fromindex[0]`
/// is initialised to `size * i` and the step function runs up to `size*(i+1)`.
///
/// # Parameters
///
/// * `tocarry`     – `n` output `Vec<i64>` buffers (one per combination column).
/// * `toindex`     – Write-position cursor, length `n` (reset to 0 on entry).
/// * `fromindex`   – Scratch current-combination buffer, length `n`.
/// * `n`           – Number of elements per combination.
/// * `replacement` – Whether to allow repeated elements.
/// * `size`        – Fixed row length of the RegularArray.
/// * `length`      – Number of rows.
///
/// # Examples
///
/// ```
/// use cpu_kernels::regular_array_combinations_64::regular_array_combinations_64;
///
/// // 2 rows of size 3; 2-combinations → (0,1),(0,2),(1,2) per row
/// let mut tocarry   = vec![vec![0i64; 6], vec![0i64; 6]];
/// let mut toindex   = vec![0usize; 2];
/// let mut fromindex = vec![0i64;  2];
/// regular_array_combinations_64(&mut tocarry, &mut toindex, &mut fromindex, 2, false, 3, 2);
/// // row 0: (0,1),(0,2),(1,2); row 1: (3,4),(3,5),(4,5)
/// assert_eq!(tocarry[0], vec![0, 0, 1, 3, 3, 4]);
/// assert_eq!(tocarry[1], vec![1, 2, 2, 4, 5, 5]);
/// ```
pub fn regular_array_combinations_64(
    tocarry: &mut Vec<Vec<i64>>,
    toindex: &mut Vec<usize>,
    fromindex: &mut Vec<i64>,
    n: usize,
    replacement: bool,
    size: i64,
    length: i64,
) {
    assert_eq!(tocarry.len(), n);
    assert_eq!(toindex.len(), n);
    assert_eq!(fromindex.len(), n);

    for v in toindex.iter_mut() {
        *v = 0;
    }

    for i in 0..length {
        let start = size * i;
        fromindex[0] = start;
        list_array_combinations_step_64(
            tocarry,
            toindex,
            fromindex,
            0,
            start + size,
            n,
            replacement,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_rows_pairs_no_replacement() {
        let mut tocarry = vec![vec![0i64; 6], vec![0i64; 6]];
        let mut toindex = vec![0usize; 2];
        let mut fromindex = vec![0i64; 2];
        regular_array_combinations_64(&mut tocarry, &mut toindex, &mut fromindex, 2, false, 3, 2);
        assert_eq!(tocarry[0], vec![0, 0, 1, 3, 3, 4]);
        assert_eq!(tocarry[1], vec![1, 2, 2, 4, 5, 5]);
    }

    #[test]
    fn one_row_pairs_with_replacement() {
        // row 0, size=3: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
        let mut tocarry = vec![vec![0i64; 6], vec![0i64; 6]];
        let mut toindex = vec![0usize; 2];
        let mut fromindex = vec![0i64; 2];
        regular_array_combinations_64(&mut tocarry, &mut toindex, &mut fromindex, 2, true, 3, 1);
        assert_eq!(tocarry[0], vec![0, 0, 0, 1, 1, 2]);
        assert_eq!(tocarry[1], vec![0, 1, 2, 1, 2, 2]);
    }

    #[test]
    fn triples_single_row() {
        // C(4,3)=4 triples in row 0 of size 4
        let mut tocarry = vec![vec![0i64; 4], vec![0i64; 4], vec![0i64; 4]];
        let mut toindex = vec![0usize; 3];
        let mut fromindex = vec![0i64; 3];
        regular_array_combinations_64(&mut tocarry, &mut toindex, &mut fromindex, 3, false, 4, 1);
        assert_eq!(tocarry[0], vec![0, 0, 0, 1]);
        assert_eq!(tocarry[1], vec![1, 1, 2, 2]);
        assert_eq!(tocarry[2], vec![2, 3, 3, 3]);
    }

    #[test]
    fn zero_length() {
        let mut tocarry = vec![vec![0i64; 0], vec![0i64; 0]];
        let mut toindex = vec![0usize; 2];
        let mut fromindex = vec![0i64; 2];
        regular_array_combinations_64(&mut tocarry, &mut toindex, &mut fromindex, 2, false, 3, 0);
        assert_eq!(toindex, [0, 0]);
    }
}
