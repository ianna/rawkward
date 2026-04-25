// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Fill an index with `0, 1, …, lenindex-1` followed by `-1`.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_numnull_unique_64.cpp`.
//!
//! This kernel initialises the output index used when building a unique
//! `IndexedOptionArray`: valid entries get sequential positions and the final
//! sentinel slot is set to `-1`.

/// Fill `toindex[0..lenindex]` with `0, 1, …, lenindex-1` and then set
/// `toindex[lenindex] = -1`.
///
/// # Panics
///
/// Panics if `toindex.len() < lenindex + 1`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_numnull_unique_64::indexed_array_numnull_unique_64;
///
/// let mut idx = [0i64; 4]; // lenindex=3, needs 4 slots
/// indexed_array_numnull_unique_64(&mut idx, 3);
/// assert_eq!(idx, [0, 1, 2, -1]);
/// ```
pub fn indexed_array_numnull_unique_64(toindex: &mut [i64], lenindex: usize) {
    assert!(
        toindex.len() >= lenindex + 1,
        "toindex must have at least lenindex+1 slots"
    );
    for i in 0..lenindex {
        toindex[i] = i as i64;
    }
    toindex[lenindex] = -1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut idx = [0i64; 4];
        indexed_array_numnull_unique_64(&mut idx, 3);
        assert_eq!(idx, [0, 1, 2, -1]);
    }

    #[test]
    fn zero_length() {
        let mut idx = [0i64; 1];
        indexed_array_numnull_unique_64(&mut idx, 0);
        assert_eq!(idx[0], -1);
    }

    #[test]
    fn single_element() {
        let mut idx = [0i64; 2];
        indexed_array_numnull_unique_64(&mut idx, 1);
        assert_eq!(idx, [0, -1]);
    }

    #[test]
    fn large() {
        let n = 100usize;
        let mut idx = vec![0i64; n + 1];
        indexed_array_numnull_unique_64(&mut idx, n);
        for i in 0..n {
            assert_eq!(idx[i], i as i64);
        }
        assert_eq!(idx[n], -1);
    }
}
