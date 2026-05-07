// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Apply a per-row advanced index to a RegularArray.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_RegularArray_getitem_next_array_advanced.cpp`.

/// For each row `i`:
/// * `tocarry   [i] = i*size + fromarray[fromadvanced[i]]`
/// * `toadvanced[i] = i`
///
/// # Examples
///
/// ```
/// use cpu_kernels::regular_array_getitem_next_array_advanced::regular_array_getitem_next_array_advanced_64;
///
/// // 3 rows of size 4; fromarray=[0,2,1]; fromadvanced=[2,0,1]
/// // row0: array[advanced[0]]=array[2]=1 → 0*4+1=1
/// // row1: array[advanced[1]]=array[0]=0 → 1*4+0=4
/// // row2: array[advanced[2]]=array[1]=2 → 2*4+2=10
/// let fromarray    = [0i64, 2, 1];
/// let fromadvanced = [2i64, 0, 1];
/// let mut carry    = [0i64; 3];
/// let mut advanced = [0i64; 3];
/// regular_array_getitem_next_array_advanced_64(&mut carry, &mut advanced, &fromadvanced, &fromarray, 3, 4);
/// assert_eq!(carry,    [1, 4, 10]);
/// assert_eq!(advanced, [0, 1, 2]);
/// ```
pub fn regular_array_getitem_next_array_advanced_64(
    tocarry: &mut [i64],
    toadvanced: &mut [i64],
    fromadvanced: &[i64],
    fromarray: &[i64],
    length: usize,
    size: i64,
) {
    assert_eq!(fromadvanced.len(), length);
    assert!(tocarry.len() >= length);
    assert!(toadvanced.len() >= length);

    for i in 0..length {
        tocarry[i] = i as i64 * size + fromarray[fromadvanced[i] as usize];
        toadvanced[i] = i as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let fromarray = [0i64, 2, 1];
        let fromadvanced = [2i64, 0, 1];
        let mut carry = [0i64; 3];
        let mut advanced = [0i64; 3];
        regular_array_getitem_next_array_advanced_64(
            &mut carry,
            &mut advanced,
            &fromadvanced,
            &fromarray,
            3,
            4,
        );
        assert_eq!(carry, [1, 4, 10]);
        assert_eq!(advanced, [0, 1, 2]);
    }

    #[test]
    fn identity() {
        let fromarray = [0i64, 1, 2];
        let fromadvanced = [0i64, 1, 2];
        let mut carry = [0i64; 3];
        let mut advanced = [0i64; 3];
        regular_array_getitem_next_array_advanced_64(
            &mut carry,
            &mut advanced,
            &fromadvanced,
            &fromarray,
            3,
            3,
        );
        assert_eq!(carry, [0, 4, 8]);
        assert_eq!(advanced, [0, 1, 2]);
    }

    #[test]
    fn single_row() {
        let fromarray = [2i64];
        let fromadvanced = [0i64];
        let mut carry = [0i64; 1];
        let mut advanced = [0i64; 1];
        regular_array_getitem_next_array_advanced_64(
            &mut carry,
            &mut advanced,
            &fromadvanced,
            &fromarray,
            1,
            5,
        );
        assert_eq!(carry[0], 2);
        assert_eq!(advanced[0], 0);
    }
}
