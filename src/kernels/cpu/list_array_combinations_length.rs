//! Compute the number of n-combinations per list and the total length.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_combinations_length.cpp`.

// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

/// For each list `i` of length `stops[i] - starts[i]`, compute the number of
/// `n`-combinations (with or without replacement) and write cumulative sums
/// into `tooffsets`.
///
/// Also returns the grand total as the first element of the returned tuple.
///
/// # Combination count formula
///
/// Without replacement: C(size, n) = size! / (n! · (size-n)!)  
/// With replacement: C(size+n-1, n)  (achieved by `size += n-1` before the calc).
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_array_combinations_length::list_array64_combinations_length_64;
///
/// // Two lists of length 3; 2-combinations of 3 = 3 each → total 6
/// let starts = [0i64, 3];
/// let stops  = [3i64, 6];
/// let mut offsets = [0i64; 3];
/// let total = list_array64_combinations_length_64(&mut offsets, 2, false, &starts, &stops);
/// assert_eq!(total, 6);
/// assert_eq!(offsets, [0, 3, 6]);
/// ```
pub fn list_array_combinations_length<C>(
    tooffsets: &mut [i64],
    n: i64,
    replacement: bool,
    starts: &[C],
    stops: &[C],
) -> i64
where
    C: Copy + Into<i64>,
{
    let length = starts.len();
    assert_eq!(stops.len(), length);
    assert!(tooffsets.len() >= length + 1);

    let mut totallen = 0i64;
    tooffsets[0] = 0;

    for i in 0..length {
        let mut size: i64 = stops[i].into() - starts[i].into();
        if replacement {
            size += n - 1;
        }
        let mut thisn = n;
        let combinationslen = if thisn > size {
            0
        } else if thisn == size {
            1
        } else {
            // Use the smaller of n and size-n to minimise iterations (C is symmetric).
            if thisn * 2 > size {
                thisn = size - thisn;
            }
            let mut c = size;
            for j in 2..=thisn {
                c *= size - j + 1;
                c /= j;
            }
            c
        };
        totallen += combinationslen;
        tooffsets[i + 1] = tooffsets[i] + combinationslen;
    }
    totallen
}

pub fn list_array32_combinations_length_64(
    tooffsets: &mut [i64],
    n: i64,
    replacement: bool,
    starts: &[i32],
    stops: &[i32],
) -> i64 {
    list_array_combinations_length(tooffsets, n, replacement, starts, stops)
}
pub fn list_array_u32_combinations_length_64(
    tooffsets: &mut [i64],
    n: i64,
    replacement: bool,
    starts: &[u32],
    stops: &[u32],
) -> i64 {
    list_array_combinations_length(tooffsets, n, replacement, starts, stops)
}
pub fn list_array64_combinations_length_64(
    tooffsets: &mut [i64],
    n: i64,
    replacement: bool,
    starts: &[i64],
    stops: &[i64],
) -> i64 {
    list_array_combinations_length(tooffsets, n, replacement, starts, stops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_no_replacement() {
        // C(3,2) = 3 per list
        let starts = [0i64, 3];
        let stops = [3i64, 6];
        let mut offsets = [0i64; 3];
        let total = list_array64_combinations_length_64(&mut offsets, 2, false, &starts, &stops);
        assert_eq!(total, 6);
        assert_eq!(offsets, [0, 3, 6]);
    }

    #[test]
    fn pairs_with_replacement() {
        // C(3+2-1, 2) = C(4,2) = 6 per list
        let starts = [0i64];
        let stops = [3i64];
        let mut offsets = [0i64; 2];
        let total = list_array64_combinations_length_64(&mut offsets, 2, true, &starts, &stops);
        assert_eq!(total, 6);
        assert_eq!(offsets, [0, 6]);
    }

    #[test]
    fn n_greater_than_size_gives_zero() {
        // n=4 > size=3 → 0 combinations
        let starts = [0i64];
        let stops = [3i64];
        let mut offsets = [0i64; 2];
        let total = list_array64_combinations_length_64(&mut offsets, 4, false, &starts, &stops);
        assert_eq!(total, 0);
    }

    #[test]
    fn n_equals_size_gives_one() {
        let starts = [0i64];
        let stops = [4i64];
        let mut offsets = [0i64; 2];
        let total = list_array64_combinations_length_64(&mut offsets, 4, false, &starts, &stops);
        assert_eq!(total, 1);
    }

    #[test]
    fn triples_no_replacement() {
        // C(5,3) = 10
        let starts = [0i64];
        let stops = [5i64];
        let mut offsets = [0i64; 2];
        let total = list_array64_combinations_length_64(&mut offsets, 3, false, &starts, &stops);
        assert_eq!(total, 10);
    }

    #[test]
    fn i32_starts_stops() {
        let starts = [0i32];
        let stops = [4i32];
        let mut offsets = [0i64; 2];
        let total = list_array32_combinations_length_64(&mut offsets, 2, false, &starts, &stops);
        assert_eq!(total, 6); // C(4,2)
    }
}
