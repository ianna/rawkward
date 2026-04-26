// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build padded offsets and total length for rpad on axis=1.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_rpad_length_axis1.cpp`.

/// For each list `i`, the new length is `max(target, fromoffsets[i+1]-fromoffsets[i])`.
/// Writes cumulative sums into `tooffsets[0..=fromlength]`.
///
/// # Returns
///
/// The total new content length.
pub fn list_offset_array_rpad_length_axis1<C>(
    tooffsets: &mut [C],
    fromoffsets: &[C],
    fromlength: usize,
    target: i64,
) -> i64
where
    C: TryFrom<i64> + Copy + Into<i64>,
    <C as TryFrom<i64>>::Error: std::fmt::Debug,
{
    assert!(tooffsets.len() > fromlength);
    tooffsets[0] = C::try_from(0i64).expect("0 fits");
    let mut length = 0i64;
    for i in 0..fromlength {
        let range: i64 = fromoffsets[i + 1].into() - fromoffsets[i].into();
        let longer = target.max(range);
        length += longer;
        let prev: i64 = tooffsets[i].into();
        tooffsets[i + 1] = C::try_from(prev + longer).expect("value fits");
    }
    length
}

pub fn list_offset_array32_rpad_length_axis1(
    tooffsets: &mut [i32],
    fromoffsets: &[i32],
    fromlength: usize,
    target: i64,
) -> i64 {
    list_offset_array_rpad_length_axis1(tooffsets, fromoffsets, fromlength, target)
}
pub fn list_offset_array64_rpad_length_axis1(
    tooffsets: &mut [i64],
    fromoffsets: &[i64],
    fromlength: usize,
    target: i64,
) -> i64 {
    list_offset_array_rpad_length_axis1(tooffsets, fromoffsets, fromlength, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pads_short_lists() {
        // lists of lengths [1, 3, 2], target=3 → [3,3,3], total=9
        let from = [0i64, 1, 4, 6];
        let mut to = [0i64; 4];
        let total = list_offset_array64_rpad_length_axis1(&mut to, &from, 3, 3);
        assert_eq!(total, 9);
        assert_eq!(to, [0, 3, 6, 9]);
    }

    #[test]
    fn no_padding_needed() {
        let from = [0i64, 5, 10];
        let mut to = [0i64; 3];
        let total = list_offset_array64_rpad_length_axis1(&mut to, &from, 2, 3);
        assert_eq!(total, 10);
        assert_eq!(to, [0, 5, 10]);
    }
}
