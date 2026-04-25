// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: count elements per group.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_count_64.cpp`.

/// Count the number of elements in each parent group.
///
/// Iterates over `parents` (length `lenparents`), incrementing
/// `toptr[parents[i]]` for each `i`.  `toptr` is zero-initialised before the
/// loop.
///
/// # Parameters
///
/// * `toptr`      – Output slice of length `outlength`; initialised to zero.
/// * `parents`    – Parent-group index for each input element.
/// * `outlength`  – Number of output groups (= `toptr.len()`).
///
/// # Panics
///
/// Panics if any `parents[i]` is out of range for `toptr`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::reduce_count_64::reduce_count_64;
///
/// let parents = [0i64, 0, 1, 2, 2, 2];
/// let mut out = [0i64; 3];
/// reduce_count_64(&mut out, &parents);
/// assert_eq!(out, [2, 1, 3]);
/// ```
pub fn reduce_count_64(toptr: &mut [i64], parents: &[i64]) {
    for v in toptr.iter_mut() {
        *v = 0;
    }
    for &p in parents {
        toptr[p as usize] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_count() {
        let parents = [0i64, 0, 1, 2, 2, 2];
        let mut out = [0i64; 3];
        reduce_count_64(&mut out, &parents);
        assert_eq!(out, [2, 1, 3]);
    }

    #[test]
    fn empty_parents() {
        let mut out = [0i64; 3];
        reduce_count_64(&mut out, &[]);
        assert_eq!(out, [0, 0, 0]);
    }

    #[test]
    fn single_group() {
        let parents = [0i64; 5];
        let mut out = [0i64; 1];
        reduce_count_64(&mut out, &parents);
        assert_eq!(out[0], 5);
    }

    #[test]
    fn output_is_zeroed_before_counting() {
        let parents = [0i64];
        let mut out = [99i64; 2]; // pre-filled with garbage
        reduce_count_64(&mut out, &parents);
        assert_eq!(out[0], 1);
        assert_eq!(out[1], 0); // must be zeroed by the kernel
    }
}
