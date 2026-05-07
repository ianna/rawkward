// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Compute compact starts/stops and total length for non-null index values within ranges.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_ranges_next_64.cpp`.

/// For each range `i` defined by `fromstarts[i]..fromstops[i]`, count the
/// non-negative (non-null) values in `index` and write:
/// * `tostarts[i]` – cumulative count before range `i`.
/// * `tostops[i]`  – cumulative count after range `i`.
///
/// Returns the total count of non-null values across all ranges.
///
/// # Panics
///
/// Panics if `fromstarts.len() != fromstops.len()` or if output slices are
/// too short.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_ranges_next_64::indexed_array64_ranges_next_64;
///
/// // Two ranges: index[0..3] = [5, -1, 6]  and  index[3..5] = [-1, 7]
/// let index      = [5i64, -1, 6, -1, 7];
/// let fromstarts = [0i64, 3];
/// let fromstops  = [3i64, 5];
/// let mut tostarts = [0i64; 2];
/// let mut tostops  = [0i64; 2];
/// let total = indexed_array64_ranges_next_64(
///     &index, &fromstarts, &fromstops, &mut tostarts, &mut tostops,
/// );
/// assert_eq!(total, 3);
/// assert_eq!(tostarts, [0, 2]);
/// assert_eq!(tostops,  [2, 3]);
/// ```
pub fn indexed_array_ranges_next_64<C>(
    index: &[C],
    fromstarts: &[i64],
    fromstops: &[i64],
    tostarts: &mut [i64],
    tostops: &mut [i64],
) -> i64
where
    C: Copy + Into<i64>,
{
    assert_eq!(fromstarts.len(), fromstops.len());
    assert!(tostarts.len() >= fromstarts.len());
    assert!(tostops.len() >= fromstarts.len());

    let mut k = 0i64;
    for i in 0..fromstarts.len() {
        let start = fromstarts[i] as usize;
        let stop = fromstops[i] as usize;
        tostarts[i] = k;
        for j in start..stop {
            let v: i64 = index[j].into();
            if v >= 0 {
                k += 1;
            }
        }
        tostops[i] = k;
    }
    k
}

pub fn indexed_array32_ranges_next_64(
    index: &[i32],
    fromstarts: &[i64],
    fromstops: &[i64],
    tostarts: &mut [i64],
    tostops: &mut [i64],
) -> i64 {
    indexed_array_ranges_next_64(index, fromstarts, fromstops, tostarts, tostops)
}
pub fn indexed_array_u32_ranges_next_64(
    index: &[u32],
    fromstarts: &[i64],
    fromstops: &[i64],
    tostarts: &mut [i64],
    tostops: &mut [i64],
) -> i64 {
    indexed_array_ranges_next_64(index, fromstarts, fromstops, tostarts, tostops)
}
pub fn indexed_array64_ranges_next_64(
    index: &[i64],
    fromstarts: &[i64],
    fromstops: &[i64],
    tostarts: &mut [i64],
    tostops: &mut [i64],
) -> i64 {
    indexed_array_ranges_next_64(index, fromstarts, fromstops, tostarts, tostops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let index = [5i64, -1, 6, -1, 7];
        let fromstarts = [0i64, 3];
        let fromstops = [3i64, 5];
        let mut ts = [0i64; 2];
        let mut tp = [0i64; 2];
        let total =
            indexed_array64_ranges_next_64(&index, &fromstarts, &fromstops, &mut ts, &mut tp);
        assert_eq!(total, 3);
        assert_eq!(ts, [0, 2]);
        assert_eq!(tp, [2, 3]);
    }

    #[test]
    fn all_null() {
        let index = [-1i64, -1];
        let fromstarts = [0i64];
        let fromstops = [2i64];
        let mut ts = [0i64; 1];
        let mut tp = [0i64; 1];
        let total =
            indexed_array64_ranges_next_64(&index, &fromstarts, &fromstops, &mut ts, &mut tp);
        assert_eq!(total, 0);
        assert_eq!(ts, [0]);
        assert_eq!(tp, [0]);
    }

    #[test]
    fn all_valid() {
        let index = [1i64, 2, 3];
        let fromstarts = [0i64];
        let fromstops = [3i64];
        let mut ts = [0i64; 1];
        let mut tp = [0i64; 1];
        let total =
            indexed_array64_ranges_next_64(&index, &fromstarts, &fromstops, &mut ts, &mut tp);
        assert_eq!(total, 3);
        assert_eq!(ts, [0]);
        assert_eq!(tp, [3]);
    }

    #[test]
    fn three_ranges_middle_empty() {
        let index = [0i64, 1, 2];
        let fromstarts = [0i64, 2, 2];
        let fromstops = [2i64, 2, 3];
        let mut ts = [0i64; 3];
        let mut tp = [0i64; 3];
        let total =
            indexed_array64_ranges_next_64(&index, &fromstarts, &fromstops, &mut ts, &mut tp);
        assert_eq!(total, 3);
        assert_eq!(ts, [0, 2, 2]);
        assert_eq!(tp, [2, 2, 3]);
    }

    #[test]
    fn i32_index() {
        let index = [0i32, -1, 2];
        let fromstarts = [0i64];
        let fromstops = [3i64];
        let mut ts = [0i64; 1];
        let mut tp = [0i64; 1];
        let total =
            indexed_array32_ranges_next_64(&index, &fromstarts, &fromstops, &mut ts, &mut tp);
        assert_eq!(total, 2);
    }
}
