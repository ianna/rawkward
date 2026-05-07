// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Count null (negative) entries in an index array.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_numnull.cpp`.

/// Count the number of negative values (nulls) in `fromindex`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_numnull::indexed_array_numnull_64;
///
/// assert_eq!(indexed_array_numnull_64(&[0i64, -1, 2, -1]), 2);
/// ```
pub fn indexed_array_numnull<C>(fromindex: &[C]) -> i64
where
    C: Copy + Into<i64>,
{
    fromindex.iter().filter(|&&v| v.into() < 0).count() as i64
}

pub fn indexed_array_numnull_32(fromindex: &[i32]) -> i64 {
    indexed_array_numnull(fromindex)
}
pub fn indexed_array_numnull_u32(fromindex: &[u32]) -> i64 {
    indexed_array_numnull(fromindex)
}
pub fn indexed_array_numnull_64(fromindex: &[i64]) -> i64 {
    indexed_array_numnull(fromindex)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed() {
        assert_eq!(indexed_array_numnull_64(&[0, -1, 2, -1]), 2);
    }
    #[test]
    fn none_null() {
        assert_eq!(indexed_array_numnull_64(&[0, 1, 2]), 0);
    }
    #[test]
    fn all_null() {
        assert_eq!(indexed_array_numnull_64(&[-1, -1]), 2);
    }
    #[test]
    fn empty() {
        assert_eq!(indexed_array_numnull_64(&[]), 0);
    }
    #[test]
    fn u32_no_negatives() {
        assert_eq!(indexed_array_numnull_u32(&[0, 1, 2]), 0);
    }
}
