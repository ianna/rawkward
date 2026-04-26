// Copyright (c) 2026 Ianna Osborne 
// SPDX-License-Identifier: BSD-3-Clause

//! Count the number of group-boundary transitions in a parents array.
//!
//! Corresponds to `src/cpu-kernels/awkward_sorting_ranges_length.cpp`.

/// Return the length needed for the `sorting_ranges` offsets array.
///
/// The result is 2 plus the number of transitions (`parents[i-1] != parents[i]`)
/// in `parents[0..parentslength]`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::sorting_ranges_length::sorting_ranges_length;
///
/// // [0,0,1,1,2] → 2 transitions → length = 4
/// let parents = [0i64, 0, 1, 1, 2];
/// assert_eq!(sorting_ranges_length(&parents), 4);
/// ```
pub fn sorting_ranges_length(parents: &[i64]) -> i64 {
    let mut length = 2i64;
    for i in 1..parents.len() {
        if parents[i - 1] != parents[i] {
            length += 1;
        }
    }
    length
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        assert_eq!(sorting_ranges_length(&[0i64, 0, 1, 1, 2]), 4);
    }

    #[test]
    fn all_same() {
        assert_eq!(sorting_ranges_length(&[0i64; 5]), 2);
    }

    #[test]
    fn all_different() {
        assert_eq!(sorting_ranges_length(&[0i64, 1, 2, 3]), 5);
    }

    #[test]
    fn single_element() {
        assert_eq!(sorting_ranges_length(&[0i64]), 2);
    }
}
