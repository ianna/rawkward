// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Spread advanced indices across the range-slice output positions.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListArray_getitem_next_range_spreadadvanced.cpp`.

/// For each list `i` (of `lenstarts` lists), fill every position in the
/// output range `fromoffsets[i]..fromoffsets[i+1]` with `fromadvanced[i]`.
///
/// ```text
/// for j in fromoffsets[i]..fromoffsets[i+1]:
///     toadvanced[j] = fromadvanced[i]
/// ```
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_array_getitem_next_range_spreadadvanced::list_array64_getitem_next_range_spreadadvanced_64;
///
/// // Two lists; offsets=[0,2,5] means 2 and 3 output elements respectively.
/// let fromadvanced = [7i64, 9];
/// let fromoffsets  = [0i64, 2, 5];
/// let mut toadvanced = [0i64; 5];
/// list_array64_getitem_next_range_spreadadvanced_64(&mut toadvanced, &fromadvanced, &fromoffsets, 2);
/// assert_eq!(toadvanced, [7, 7, 9, 9, 9]);
/// ```
pub fn list_array_getitem_next_range_spreadadvanced<C>(
    toadvanced: &mut [i64],
    fromadvanced: &[i64],
    fromoffsets: &[C],
    lenstarts: usize,
) where
    C: Copy + Into<i64>,
{
    assert!(fromoffsets.len() > lenstarts);
    assert_eq!(fromadvanced.len(), lenstarts);

    for i in 0..lenstarts {
        let start: i64 = fromoffsets[i].into();
        let stop: i64 = fromoffsets[i + 1].into();
        let count = (stop - start) as usize;
        for j in 0..count {
            toadvanced[start as usize + j] = fromadvanced[i];
        }
    }
}

pub fn list_array32_getitem_next_range_spreadadvanced_64(
    toadvanced: &mut [i64],
    fromadvanced: &[i64],
    fromoffsets: &[i32],
    lenstarts: usize,
) {
    list_array_getitem_next_range_spreadadvanced(toadvanced, fromadvanced, fromoffsets, lenstarts);
}
pub fn list_array_u32_getitem_next_range_spreadadvanced_64(
    toadvanced: &mut [i64],
    fromadvanced: &[i64],
    fromoffsets: &[u32],
    lenstarts: usize,
) {
    list_array_getitem_next_range_spreadadvanced(toadvanced, fromadvanced, fromoffsets, lenstarts);
}
pub fn list_array64_getitem_next_range_spreadadvanced_64(
    toadvanced: &mut [i64],
    fromadvanced: &[i64],
    fromoffsets: &[i64],
    lenstarts: usize,
) {
    list_array_getitem_next_range_spreadadvanced(toadvanced, fromadvanced, fromoffsets, lenstarts);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let fromadvanced = [7i64, 9];
        let fromoffsets = [0i64, 2, 5];
        let mut toadvanced = [0i64; 5];
        list_array64_getitem_next_range_spreadadvanced_64(
            &mut toadvanced,
            &fromadvanced,
            &fromoffsets,
            2,
        );
        assert_eq!(toadvanced, [7, 7, 9, 9, 9]);
    }

    #[test]
    fn single_list() {
        let fromadvanced = [3i64];
        let fromoffsets = [0i64, 4];
        let mut toadvanced = [0i64; 4];
        list_array64_getitem_next_range_spreadadvanced_64(
            &mut toadvanced,
            &fromadvanced,
            &fromoffsets,
            1,
        );
        assert_eq!(toadvanced, [3, 3, 3, 3]);
    }

    #[test]
    fn empty_range() {
        // One list with zero-length range
        let fromadvanced = [5i64];
        let fromoffsets = [2i64, 2];
        let mut toadvanced = [0i64; 0];
        list_array64_getitem_next_range_spreadadvanced_64(
            &mut toadvanced,
            &fromadvanced,
            &fromoffsets,
            1,
        );
        // Nothing written, no panic.
    }

    #[test]
    fn i32_offsets() {
        let fromadvanced = [1i64, 2];
        let fromoffsets = [0i32, 1, 3];
        let mut toadvanced = [0i64; 3];
        list_array32_getitem_next_range_spreadadvanced_64(
            &mut toadvanced,
            &fromadvanced,
            &fromoffsets,
            2,
        );
        assert_eq!(toadvanced, [1, 2, 2]);
    }
}
