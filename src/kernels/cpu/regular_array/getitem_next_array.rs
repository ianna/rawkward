// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Apply an index array to every row of a RegularArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_RegularArray_getitem_next_array.cpp`.

/// For each row `i` and each index `j` in `fromarray[0..lenarray]`:
/// * `tocarry   [i*lenarray + j] = i*size + fromarray[j]`
/// * `toadvanced[i*lenarray + j] = j`
///
/// `fromarray` values are already regularised (non-negative, in `0..size`).
/// Use [`regular_array_getitem_next_array_regularize_64`] first if needed.
///
/// # Examples
///
/// ```
/// use cpu_kernels::regular_array_getitem_next_array::regular_array_getitem_next_array_64;
///
/// // 2 rows of size 4; fromarray=[1,3]
/// let fromarray = [1i64, 3];
/// let mut carry    = [0i64; 4];
/// let mut advanced = [0i64; 4];
/// regular_array_getitem_next_array_64(&mut carry, &mut advanced, &fromarray, 2, 4);
/// assert_eq!(carry,    [1, 3, 5, 7]);  // row0: 0*4+1, 0*4+3; row1: 1*4+1, 1*4+3
/// assert_eq!(advanced, [0, 1, 0, 1]);
/// ```
pub fn regular_array_getitem_next_array_64(
    tocarry: &mut [i64],
    toadvanced: &mut [i64],
    fromarray: &[i64],
    length: usize,
    size: i64,
) {
    let lenarray = fromarray.len();
    assert!(tocarry.len() >= length * lenarray);
    assert!(toadvanced.len() >= length * lenarray);

    for i in 0..length {
        for j in 0..lenarray {
            tocarry[i * lenarray + j] = i as i64 * size + fromarray[j];
            toadvanced[i * lenarray + j] = j as i64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let fromarray = [1i64, 3];
        let mut carry = [0i64; 4];
        let mut advanced = [0i64; 4];
        regular_array_getitem_next_array_64(&mut carry, &mut advanced, &fromarray, 2, 4);
        assert_eq!(carry, [1, 3, 5, 7]);
        assert_eq!(advanced, [0, 1, 0, 1]);
    }

    #[test]
    fn single_index() {
        let fromarray = [0i64];
        let mut carry = [0i64; 3];
        let mut advanced = [0i64; 3];
        regular_array_getitem_next_array_64(&mut carry, &mut advanced, &fromarray, 3, 5);
        assert_eq!(carry, [0, 5, 10]);
        assert_eq!(advanced, [0, 0, 0]);
    }

    #[test]
    fn zero_length() {
        let fromarray = [1i64];
        let mut carry: [i64; 0] = [];
        let mut advanced: [i64; 0] = [];
        regular_array_getitem_next_array_64(&mut carry, &mut advanced, &fromarray, 0, 3);
    }
}
