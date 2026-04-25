// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Argsort an index array by value, writing the sorted positions to carry.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_local_preparenext_64.cpp`.

/// Fill `tocarry` with the permutation that would sort `fromindex` in
/// ascending order.
///
/// This is equivalent to `argsort(fromindex)` — position `i` in `tocarry`
/// holds the original index whose value belongs at position `i` when
/// `fromindex` is sorted.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_offset_array_local_preparenext_64::list_offset_array_local_preparenext_64;
///
/// let fromindex = [30i64, 10, 20];
/// let mut carry = [0i64; 3];
/// list_offset_array_local_preparenext_64(&mut carry, &fromindex);
/// assert_eq!(carry, [1, 2, 0]); // index[1]=10 < index[2]=20 < index[0]=30
/// ```
pub fn list_offset_array_local_preparenext_64(tocarry: &mut [i64], fromindex: &[i64]) {
    let length = fromindex.len();
    assert_eq!(tocarry.len(), length);

    let mut result: Vec<usize> = (0..length).collect();
    result.sort_by_key(|&i| fromindex[i]);

    for (i, &r) in result.iter().enumerate() {
        tocarry[i] = r as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let fromindex = [30i64, 10, 20];
        let mut carry = [0i64; 3];
        list_offset_array_local_preparenext_64(&mut carry, &fromindex);
        assert_eq!(carry, [1, 2, 0]);
    }

    #[test]
    fn already_sorted() {
        let fromindex = [1i64, 2, 3, 4];
        let mut carry = [0i64; 4];
        list_offset_array_local_preparenext_64(&mut carry, &fromindex);
        assert_eq!(carry, [0, 1, 2, 3]);
    }

    #[test]
    fn reverse_sorted() {
        let fromindex = [4i64, 3, 2, 1];
        let mut carry = [0i64; 4];
        list_offset_array_local_preparenext_64(&mut carry, &fromindex);
        assert_eq!(carry, [3, 2, 1, 0]);
    }

    #[test]
    fn single_element() {
        let fromindex = [42i64];
        let mut carry = [0i64; 1];
        list_offset_array_local_preparenext_64(&mut carry, &fromindex);
        assert_eq!(carry, [0]);
    }

    #[test]
    fn stable_equal_values() {
        // sort_by_key is stable, so equal values keep original order
        let fromindex = [5i64, 5, 5];
        let mut carry = [0i64; 3];
        list_offset_array_local_preparenext_64(&mut carry, &fromindex);
        assert_eq!(carry, [0, 1, 2]);
    }
}
