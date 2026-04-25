// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Spread per-row advanced indices across the range-slice output positions.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_RegularArray_getitem_next_range_spreadadvanced.cpp`.

/// For each row `i` and slot `j` in `0..nextsize`:
/// `toadvanced[i*nextsize + j] = fromadvanced[i]`
///
/// # Examples
///
/// ```
/// use cpu_kernels::regular_array_getitem_next_range_spreadadvanced::regular_array_getitem_next_range_spreadadvanced_64;
///
/// // 2 rows; nextsize=3; fromadvanced=[7, 9]
/// let fromadvanced = [7i64, 9];
/// let mut toadvanced = [0i64; 6];
/// regular_array_getitem_next_range_spreadadvanced_64(&mut toadvanced, &fromadvanced, 2, 3);
/// assert_eq!(toadvanced, [7, 7, 7, 9, 9, 9]);
/// ```
pub fn regular_array_getitem_next_range_spreadadvanced_64(
    toadvanced: &mut [i64],
    fromadvanced: &[i64],
    length: usize,
    nextsize: usize,
) {
    assert_eq!(fromadvanced.len(), length);
    assert!(toadvanced.len() >= length * nextsize);

    for i in 0..length {
        for j in 0..nextsize {
            toadvanced[i * nextsize + j] = fromadvanced[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let fromadvanced = [7i64, 9];
        let mut toadvanced = [0i64; 6];
        regular_array_getitem_next_range_spreadadvanced_64(&mut toadvanced, &fromadvanced, 2, 3);
        assert_eq!(toadvanced, [7, 7, 7, 9, 9, 9]);
    }

    #[test]
    fn single_row() {
        let fromadvanced = [42i64];
        let mut toadvanced = [0i64; 4];
        regular_array_getitem_next_range_spreadadvanced_64(&mut toadvanced, &fromadvanced, 1, 4);
        assert_eq!(toadvanced, [42, 42, 42, 42]);
    }

    #[test]
    fn nextsize_one() {
        let fromadvanced = [1i64, 3, 5];
        let mut toadvanced = [0i64; 3];
        regular_array_getitem_next_range_spreadadvanced_64(&mut toadvanced, &fromadvanced, 3, 1);
        assert_eq!(toadvanced, [1, 3, 5]);
    }

    #[test]
    fn zero_length() {
        let fromadvanced: [i64; 0] = [];
        let mut toadvanced: [i64; 0] = [];
        regular_array_getitem_next_range_spreadadvanced_64(&mut toadvanced, &fromadvanced, 0, 3);
    }
}
