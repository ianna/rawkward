// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Adjust offsets by subtracting the cumulative count of null indexes seen so far.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_drop_none_indexes.cpp`.

/// For each offset position `i` in `fromoffsets[0..length_offsets]`, subtract
/// the number of negative entries in `noneindexes[0..fromoffsets[i]]` and
/// write the result into `tooffsets[i]`.
///
/// This effectively re-aligns offsets after null entries have been removed
/// from the flat content index array.
///
/// # Parameters
///
/// * `tooffsets`      – Output offsets; same length as `fromoffsets`.
/// * `noneindexes`    – Flat index array; negative values are nulls.
/// * `fromoffsets`    – Input offsets (length `length_offsets`).
/// * `length_offsets` – Number of entries to process.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_offset_array_drop_none_indexes::list_offset_array_drop_none_indexes_64;
///
/// // noneindexes=[0, -1, 2, -1, 4]; fromoffsets=[0, 2, 5]
/// // At offset[0]=0: 0 nones seen → 0-0=0
/// // At offset[1]=2: 1 none in [0..2] → 2-1=1
/// // At offset[2]=5: 2 nones in [0..5] → 5-2=3
/// let noneindexes = [0i64, -1, 2, -1, 4];
/// let fromoffsets = [0i64, 2, 5];
/// let mut tooffsets = [0i64; 3];
/// list_offset_array_drop_none_indexes_64(&mut tooffsets, &noneindexes, &fromoffsets, 3);
/// assert_eq!(tooffsets, [0, 1, 3]);
/// ```
pub fn list_offset_array_drop_none_indexes<T>(
    tooffsets: &mut [T],
    noneindexes: &[T],
    fromoffsets: &[T],
    length_offsets: usize,
) where
    T: Copy + Into<i64> + TryFrom<i64>,
    <T as TryFrom<i64>>::Error: std::fmt::Debug,
{
    assert!(tooffsets.len() >= length_offsets);
    assert!(fromoffsets.len() >= length_offsets);

    let mut nr_of_nones = 0i64;
    let mut offset1 = 0i64;

    for i in 0..length_offsets {
        let offset2: i64 = fromoffsets[i].into();
        for j in offset1..offset2 {
            if noneindexes[j as usize].into() < 0 {
                nr_of_nones += 1;
            }
        }
        let raw: i64 = fromoffsets[i].into();
        tooffsets[i] = T::try_from(raw - nr_of_nones).expect("value fits");
        offset1 = offset2;
    }
}

pub fn list_offset_array_drop_none_indexes_64(
    tooffsets: &mut [i64],
    noneindexes: &[i64],
    fromoffsets: &[i64],
    length_offsets: usize,
) {
    list_offset_array_drop_none_indexes(tooffsets, noneindexes, fromoffsets, length_offsets);
}
pub fn list_offset_array_drop_none_indexes_32(
    tooffsets: &mut [i32],
    noneindexes: &[i32],
    fromoffsets: &[i32],
    length_offsets: usize,
) {
    list_offset_array_drop_none_indexes(tooffsets, noneindexes, fromoffsets, length_offsets);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let noneindexes = [0i64, -1, 2, -1, 4];
        let fromoffsets = [0i64, 2, 5];
        let mut tooffsets = [0i64; 3];
        list_offset_array_drop_none_indexes_64(&mut tooffsets, &noneindexes, &fromoffsets, 3);
        assert_eq!(tooffsets, [0, 1, 3]);
    }

    #[test]
    fn no_nones() {
        let noneindexes = [0i64, 1, 2, 3];
        let fromoffsets = [0i64, 2, 4];
        let mut tooffsets = [0i64; 3];
        list_offset_array_drop_none_indexes_64(&mut tooffsets, &noneindexes, &fromoffsets, 3);
        assert_eq!(tooffsets, [0, 2, 4]);
    }

    #[test]
    fn all_nones() {
        let noneindexes = [-1i64, -1, -1];
        let fromoffsets = [0i64, 2, 3];
        let mut tooffsets = [0i64; 3];
        list_offset_array_drop_none_indexes_64(&mut tooffsets, &noneindexes, &fromoffsets, 3);
        assert_eq!(tooffsets, [0, 0, 0]);
    }

    #[test]
    fn i32_variant() {
        let noneindexes = [0i32, -1, 2];
        let fromoffsets = [0i32, 2, 3];
        let mut tooffsets = [0i32; 3];
        list_offset_array_drop_none_indexes_32(&mut tooffsets, &noneindexes, &fromoffsets, 3);
        assert_eq!(tooffsets, [0, 1, 2]);
    }
}
