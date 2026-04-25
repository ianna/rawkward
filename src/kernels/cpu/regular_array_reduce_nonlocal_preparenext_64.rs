// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build carry and parents for a non-local reduction over a RegularArray.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_RegularArray_reduce_nonlocal_preparenext_64.cpp`.

/// Populate `nextcarry` and `nextparents` by iterating in *column-major* order
/// (j outer, i inner), ensuring `nextparents` is sorted (locally contiguous):
///
/// For `j` in `0..size`, then `i` in `0..length`:
/// * `nextcarry  [k] = i*size + j`
/// * `nextparents[k] = parents[i]*size + j`
/// * `k++`
///
/// # Examples
///
/// ```
/// use cpu_kernels::regular_array_reduce_nonlocal_preparenext_64::regular_array_reduce_nonlocal_preparenext_64;
///
/// // 3 rows of size 2; parents=[0,0,1]
/// let parents = [0i64, 0, 1];
/// let mut carry   = [0i64; 6];
/// let mut np      = [0i64; 6];
/// regular_array_reduce_nonlocal_preparenext_64(&mut carry, &mut np, &parents, 2, 3);
/// // j=0: i=0→(0,0), i=1→(2,0), i=2→(4,2)
/// // j=1: i=0→(1,1), i=1→(3,1), i=2→(5,3)
/// assert_eq!(carry, [0, 2, 4, 1, 3, 5]);
/// assert_eq!(np,    [0, 0, 2, 1, 1, 3]);
/// ```
pub fn regular_array_reduce_nonlocal_preparenext_64(
    nextcarry: &mut [i64],
    nextparents: &mut [i64],
    parents: &[i64],
    size: usize,
    length: usize,
) {
    assert_eq!(parents.len(), length);
    assert!(nextcarry.len() >= length * size);
    assert!(nextparents.len() >= length * size);

    let mut k = 0usize;
    for j in 0..size {
        for i in 0..length {
            nextcarry[k] = (i * size + j) as i64;
            nextparents[k] = parents[i] * size as i64 + j as i64;
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let parents = [0i64, 0, 1];
        let mut carry = [0i64; 6];
        let mut np = [0i64; 6];
        regular_array_reduce_nonlocal_preparenext_64(&mut carry, &mut np, &parents, 2, 3);
        assert_eq!(carry, [0, 2, 4, 1, 3, 5]);
        assert_eq!(np, [0, 0, 2, 1, 1, 3]);
    }

    #[test]
    fn single_row_single_column() {
        let parents = [0i64];
        let mut carry = [0i64; 1];
        let mut np = [0i64; 1];
        regular_array_reduce_nonlocal_preparenext_64(&mut carry, &mut np, &parents, 1, 1);
        assert_eq!(carry, [0]);
        assert_eq!(np, [0]);
    }

    #[test]
    fn all_same_parent() {
        let parents = [0i64; 4];
        let mut carry = [0i64; 8];
        let mut np = [0i64; 8];
        regular_array_reduce_nonlocal_preparenext_64(&mut carry, &mut np, &parents, 2, 4);
        // j=0: 0,2,4,6; j=1: 1,3,5,7
        assert_eq!(carry, [0, 2, 4, 6, 1, 3, 5, 7]);
        assert_eq!(np, [0, 0, 0, 0, 1, 1, 1, 1]);
    }
}
