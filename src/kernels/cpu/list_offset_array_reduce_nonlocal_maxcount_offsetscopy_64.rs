// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Compute the maximum list length and copy offsets.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_maxcount_offsetscopy_64.cpp`.

/// Scan `offsets[0..=length]`, compute the maximum `offsets[i+1]-offsets[i]`
/// (stored in `*maxcount`), and copy `offsets` verbatim into `offsetscopy`.
///
/// # Returns
///
/// `(maxcount, ())` — `maxcount` is also written through `out_maxcount`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_offset_array_reduce_nonlocal_maxcount_offsetscopy_64::list_offset_array_reduce_nonlocal_maxcount_offsetscopy_64;
///
/// let offsets = [0i64, 3, 5, 9];
/// let mut offsetscopy = [0i64; 4];
/// let maxcount = list_offset_array_reduce_nonlocal_maxcount_offsetscopy_64(
///     &mut offsetscopy, &offsets, 3,
/// );
/// assert_eq!(maxcount, 4);
/// assert_eq!(offsetscopy, offsets);
/// ```
pub fn list_offset_array_reduce_nonlocal_maxcount_offsetscopy_64(
    offsetscopy: &mut [i64],
    offsets: &[i64],
    length: usize,
) -> i64 {
    assert!(offsets.len() >= length + 1);
    assert!(offsetscopy.len() >= length + 1);

    let mut maxcount = 0i64;
    offsetscopy[0] = offsets[0];
    for i in 0..length {
        let count = offsets[i + 1] - offsets[i];
        if count > maxcount {
            maxcount = count;
        }
        offsetscopy[i + 1] = offsets[i + 1];
    }
    maxcount
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let offsets = [0i64, 3, 5, 9];
        let mut copy = [0i64; 4];
        let m = list_offset_array_reduce_nonlocal_maxcount_offsetscopy_64(&mut copy, &offsets, 3);
        assert_eq!(m, 4);
        assert_eq!(copy, offsets);
    }

    #[test]
    fn all_same_length() {
        let offsets = [0i64, 2, 4, 6];
        let mut copy = [0i64; 4];
        let m = list_offset_array_reduce_nonlocal_maxcount_offsetscopy_64(&mut copy, &offsets, 3);
        assert_eq!(m, 2);
    }

    #[test]
    fn single_list() {
        let offsets = [0i64, 7];
        let mut copy = [0i64; 2];
        let m = list_offset_array_reduce_nonlocal_maxcount_offsetscopy_64(&mut copy, &offsets, 1);
        assert_eq!(m, 7);
        assert_eq!(copy, offsets);
    }

    #[test]
    fn empty_lists() {
        let offsets = [0i64, 0, 0];
        let mut copy = [0i64; 3];
        let m = list_offset_array_reduce_nonlocal_maxcount_offsetscopy_64(&mut copy, &offsets, 2);
        assert_eq!(m, 0);
    }
}
