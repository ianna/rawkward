// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Per-element null flag and total null count for an indexed array.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_numnull_parents.cpp`.

/// For each `i`, set `numnull[i] = 1` if `fromindex[i] < 0`, else `0`.
/// Also accumulate the total count into `tolength`.
///
/// # Returns `(numnull_written, total_nulls)`
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_numnull_parents::indexed_array_numnull_parents_64;
///
/// let from = [0i64, -1, 2, -1];
/// let mut numnull = [0i64; 4];
/// let total = indexed_array_numnull_parents_64(&mut numnull, &from);
/// assert_eq!(numnull, [0, 1, 0, 1]);
/// assert_eq!(total, 2);
/// ```
pub fn indexed_array_numnull_parents<C>(numnull: &mut [i64], fromindex: &[C]) -> i64
where
    C: Copy + Into<i64>,
{
    assert_eq!(numnull.len(), fromindex.len());
    let mut total = 0i64;
    for (n, &v) in numnull.iter_mut().zip(fromindex.iter()) {
        if v.into() < 0 {
            *n = 1;
            total += 1;
        } else {
            *n = 0;
        }
    }
    total
}

pub fn indexed_array_numnull_parents_32(numnull: &mut [i64], fromindex: &[i32]) -> i64 {
    indexed_array_numnull_parents(numnull, fromindex)
}
pub fn indexed_array_numnull_parents_u32(numnull: &mut [i64], fromindex: &[u32]) -> i64 {
    indexed_array_numnull_parents(numnull, fromindex)
}
pub fn indexed_array_numnull_parents_64(numnull: &mut [i64], fromindex: &[i64]) -> i64 {
    indexed_array_numnull_parents(numnull, fromindex)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed() {
        let from = [0i64, -1, 2, -1];
        let mut n = [0i64; 4];
        let t = indexed_array_numnull_parents_64(&mut n, &from);
        assert_eq!(n, [0, 1, 0, 1]);
        assert_eq!(t, 2);
    }

    #[test]
    fn none_null() {
        let from = [0i64, 1, 2];
        let mut n = [0i64; 3];
        let t = indexed_array_numnull_parents_64(&mut n, &from);
        assert_eq!(n, [0, 0, 0]);
        assert_eq!(t, 0);
    }

    #[test]
    fn all_null() {
        let from = [-1i64, -1];
        let mut n = [0i64; 2];
        let t = indexed_array_numnull_parents_64(&mut n, &from);
        assert_eq!(n, [1, 1]);
        assert_eq!(t, 2);
    }
}
