// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build a compact offsets array from a parents array for sorting.
//!
//! Corresponds to `src/cpu-kernels/awkward_sorting_ranges.cpp`.

/// Fill `toindex` with group-start positions derived from `parents`.
///
/// `toindex[0] = 0`; for each transition `parents[i-1] != parents[i]`,
/// `toindex[j++] = i`; `toindex[tolength-1] = parentslength`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::sorting_ranges::sorting_ranges;
///
/// // parents=[0,0,1,1,2]; tolength=4
/// let parents = [0i64, 0, 1, 1, 2];
/// let mut out = [0i64; 4];
/// sorting_ranges(&mut out, 4, &parents);
/// assert_eq!(out, [0, 2, 4, 5]);
/// ```
pub fn sorting_ranges(toindex: &mut [i64], tolength: usize, parents: &[i64]) {
    let parentslength = parents.len();
    assert!(toindex.len() >= tolength);

    let mut j = 0usize;
    let mut k = 0i64;
    toindex[0] = k;
    k += 1;
    j += 1;
    for i in 1..parentslength {
        if parents[i - 1] != parents[i] {
            toindex[j] = k;
            j += 1;
        }
        k += 1;
    }
    if tolength > 0 {
        toindex[tolength - 1] = parentslength as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let parents = [0i64, 0, 1, 1, 2];
        let mut out = [0i64; 4];
        sorting_ranges(&mut out, 4, &parents);
        assert_eq!(out, [0, 2, 4, 5]);
    }

    #[test]
    fn all_same() {
        let parents = [0i64; 4];
        let mut out = [0i64; 2];
        sorting_ranges(&mut out, 2, &parents);
        assert_eq!(out, [0, 4]);
    }

    #[test]
    fn all_different() {
        let parents = [0i64, 1, 2];
        let mut out = [0i64; 4];
        sorting_ranges(&mut out, 4, &parents);
        assert_eq!(out, [0, 1, 2, 3]);
    }
}
