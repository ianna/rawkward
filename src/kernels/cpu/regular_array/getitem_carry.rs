// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Expand a carry index to cover all elements within each carried row.
//!
//! Corresponds to `src/cpu-kernels/awkward_RegularArray_getitem_carry.cpp`.

/// For each element `i` in `fromcarry`, write `size` consecutive carry values:
/// `tocarry[i*size + j] = fromcarry[i] * size + j`  for `j` in `0..size`.
///
/// This maps a "row carry" (selecting which rows to keep) into a "flat carry"
/// covering all elements within each selected row.
///
/// # Examples
///
/// ```
/// use cpu_kernels::regular_array_getitem_carry::regular_array_getitem_carry_64;
///
/// // Carry rows [1, 0]; size=3 → content positions [3,4,5, 0,1,2]
/// let fromcarry = [1i64, 0];
/// let mut tocarry = [0i64; 6];
/// regular_array_getitem_carry_64(&mut tocarry, &fromcarry, 3);
/// assert_eq!(tocarry, [3, 4, 5, 0, 1, 2]);
/// ```
pub fn regular_array_getitem_carry_64(tocarry: &mut [i64], fromcarry: &[i64], size: i64) {
    let lencarry = fromcarry.len();
    assert!(tocarry.len() >= lencarry * size as usize);
    for i in 0..lencarry {
        for j in 0..size as usize {
            tocarry[i * size as usize + j] = fromcarry[i] * size + j as i64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let fromcarry = [1i64, 0];
        let mut tocarry = [0i64; 6];
        regular_array_getitem_carry_64(&mut tocarry, &fromcarry, 3);
        assert_eq!(tocarry, [3, 4, 5, 0, 1, 2]);
    }

    #[test]
    fn single_row() {
        let fromcarry = [2i64];
        let mut tocarry = [0i64; 4];
        regular_array_getitem_carry_64(&mut tocarry, &fromcarry, 4);
        assert_eq!(tocarry, [8, 9, 10, 11]);
    }

    #[test]
    fn size_one() {
        let fromcarry = [0i64, 2, 4];
        let mut tocarry = [0i64; 3];
        regular_array_getitem_carry_64(&mut tocarry, &fromcarry, 1);
        assert_eq!(tocarry, [0, 2, 4]);
    }

    #[test]
    fn identity_carry() {
        let fromcarry = [0i64, 1, 2];
        let mut tocarry = [0i64; 6];
        regular_array_getitem_carry_64(&mut tocarry, &fromcarry, 2);
        assert_eq!(tocarry, [0, 1, 2, 3, 4, 5]);
    }
}
