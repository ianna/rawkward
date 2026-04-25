// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Tile a single jagged-offset template across all rows of a RegularArray.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_RegularArray_getitem_jagged_expand.cpp`.

/// For each row `i` of `regularlength` rows and each slot `j` in
/// `0..regularsize`, tile the `singleoffsets` template:
///
/// * `multistarts[i*regularsize + j] = singleoffsets[j]`
/// * `multistops [i*regularsize + j] = singleoffsets[j+1]`
///
/// # Examples
///
/// ```
/// use cpu_kernels::regular_array_getitem_jagged_expand::regular_array_getitem_jagged_expand_64;
///
/// // regularsize=2, regularlength=3; singleoffsets=[0,3,7]
/// let singleoffsets = [0i64, 3, 7];
/// let mut ms = [0i64; 6];
/// let mut mp = [0i64; 6];
/// regular_array_getitem_jagged_expand_64(&mut ms, &mut mp, &singleoffsets, 2, 3);
/// assert_eq!(ms, [0, 3, 0, 3, 0, 3]);
/// assert_eq!(mp, [3, 7, 3, 7, 3, 7]);
/// ```
pub fn regular_array_getitem_jagged_expand_64(
    multistarts: &mut [i64],
    multistops: &mut [i64],
    singleoffsets: &[i64],
    regularsize: usize,
    regularlength: usize,
) {
    assert!(singleoffsets.len() >= regularsize + 1);
    assert!(multistarts.len() >= regularlength * regularsize);
    assert!(multistops.len() >= regularlength * regularsize);

    for i in 0..regularlength {
        for j in 0..regularsize {
            multistarts[i * regularsize + j] = singleoffsets[j];
            multistops[i * regularsize + j] = singleoffsets[j + 1];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let singleoffsets = [0i64, 3, 7];
        let mut ms = [0i64; 6];
        let mut mp = [0i64; 6];
        regular_array_getitem_jagged_expand_64(&mut ms, &mut mp, &singleoffsets, 2, 3);
        assert_eq!(ms, [0, 3, 0, 3, 0, 3]);
        assert_eq!(mp, [3, 7, 3, 7, 3, 7]);
    }

    #[test]
    fn single_row() {
        let singleoffsets = [0i64, 2, 5, 9];
        let mut ms = [0i64; 3];
        let mut mp = [0i64; 3];
        regular_array_getitem_jagged_expand_64(&mut ms, &mut mp, &singleoffsets, 3, 1);
        assert_eq!(ms, [0, 2, 5]);
        assert_eq!(mp, [2, 5, 9]);
    }

    #[test]
    fn zero_length() {
        let singleoffsets = [0i64, 1];
        let mut ms: [i64; 0] = [];
        let mut mp: [i64; 0] = [];
        regular_array_getitem_jagged_expand_64(&mut ms, &mut mp, &singleoffsets, 1, 0);
    }
}
