// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Replace `-1` (None) entries in an index with unique, monotonically
//! increasing values beyond the existing non-null entries.
//!
//! Corresponds to `src/cpu-kernels/awkward_Index_nones_as_index.cpp`.

/// Replace every `-1` in `toindex` with a unique index that extends beyond
/// the existing non-null indices.
///
/// The array is assumed to contain unique, contiguous, zero-based integers or
/// `-1`.  The kernel first counts `n_non_null` valid entries, then rewrites
/// each `-1` as `n_non_null, n_non_null+1, …` in scan order.
///
/// # Examples
///
/// ```
/// use cpu_kernels::index_nones_as_index::index_nones_as_index_64;
///
/// let mut idx = [0i64, -1, 2, -1];
/// index_nones_as_index_64(&mut idx);
/// // 2 non-null entries → nones become 3, 4 (0-based from 3)
/// assert_eq!(idx, [0, 2, 2, 3]);
/// ```
pub fn index_nones_as_index(toindex: &mut [i64]) {
    let n_non_null = toindex.iter().filter(|&&v| v != -1).count() as i64;
    let mut next = n_non_null;
    for v in toindex.iter_mut() {
        if *v == -1 {
            *v = next;
            next += 1;
        }
    }
}

/// Typed alias matching `awkward_Index_nones_as_index_64`.
pub fn index_nones_as_index_64(toindex: &mut [i64]) {
    index_nones_as_index(toindex);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_nones() {
        let mut idx = [0i64, 1, 2];
        index_nones_as_index_64(&mut idx);
        assert_eq!(idx, [0, 1, 2]);
    }

    #[test]
    fn all_nones() {
        let mut idx = [-1i64, -1, -1];
        index_nones_as_index_64(&mut idx);
        assert_eq!(idx, [0, 1, 2]);
    }

    #[test]
    fn mixed() {
        let mut idx = [0i64, -1, 2, -1];
        index_nones_as_index_64(&mut idx);
        assert_eq!(idx, [0, 2, 2, 3]);
    }

    #[test]
    fn single_none() {
        let mut idx = [0i64, 1, -1];
        index_nones_as_index_64(&mut idx);
        assert_eq!(idx, [0, 1, 2]);
    }
}
