// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Fill a flat index with within-row positions for a RegularArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_RegularArray_localindex.cpp`.

/// For each row `i` and slot `j` in `0..size`:
/// `toindex[i*size + j] = j`
///
/// The result is the local (within-row) position index for every element of a
/// RegularArray.
///
/// # Examples
///
/// ```
/// use cpu_kernels::regular_array_localindex::regular_array_localindex_64;
///
/// // 2 rows of size 3
/// let mut toindex = [0i64; 6];
/// regular_array_localindex_64(&mut toindex, 3, 2);
/// assert_eq!(toindex, [0, 1, 2, 0, 1, 2]);
/// ```
pub fn regular_array_localindex_64(toindex: &mut [i64], size: usize, length: usize) {
    assert!(toindex.len() >= length * size);
    for i in 0..length {
        for j in 0..size {
            toindex[i * size + j] = j as i64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut toindex = [0i64; 6];
        regular_array_localindex_64(&mut toindex, 3, 2);
        assert_eq!(toindex, [0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn size_one() {
        let mut toindex = [0i64; 4];
        regular_array_localindex_64(&mut toindex, 1, 4);
        assert_eq!(toindex, [0, 0, 0, 0]);
    }

    #[test]
    fn single_row() {
        let mut toindex = [0i64; 5];
        regular_array_localindex_64(&mut toindex, 5, 1);
        assert_eq!(toindex, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn zero_length() {
        let mut toindex: [i64; 0] = [];
        regular_array_localindex_64(&mut toindex, 3, 0);
    }
}
