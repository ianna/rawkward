// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Compute the total output length for rpad-and-clip on axis=1.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListArray_rpad_and_clip_length_axis1.cpp`.

/// For each list `i`, the output length is `max(target, stops[i]-starts[i])`.
/// Returns the sum.
pub fn list_array_rpad_and_clip_length_axis1<C>(
    fromstarts: &[C],
    fromstops: &[C],
    target: i64,
) -> i64
where
    C: Copy + Into<i64>,
{
    fromstarts
        .iter()
        .zip(fromstops.iter())
        .map(|(&s, &e)| {
            let range: i64 = e.into() - s.into();
            target.max(range)
        })
        .sum()
}

pub fn list_array32_rpad_and_clip_length_axis1(
    fromstarts: &[i32],
    fromstops: &[i32],
    target: i64,
) -> i64 {
    list_array_rpad_and_clip_length_axis1(fromstarts, fromstops, target)
}
pub fn list_array_u32_rpad_and_clip_length_axis1(
    fromstarts: &[u32],
    fromstops: &[u32],
    target: i64,
) -> i64 {
    list_array_rpad_and_clip_length_axis1(fromstarts, fromstops, target)
}
pub fn list_array64_rpad_and_clip_length_axis1(
    fromstarts: &[i64],
    fromstops: &[i64],
    target: i64,
) -> i64 {
    list_array_rpad_and_clip_length_axis1(fromstarts, fromstops, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_needed() {
        // lists of length 1, 5, 2; target=3 → max(3,1)+max(3,5)+max(3,2) = 3+5+3 = 11
        assert_eq!(
            list_array64_rpad_and_clip_length_axis1(&[0, 1, 6], &[1, 6, 8], 3),
            11
        );
    }

    #[test]
    fn no_padding_needed() {
        assert_eq!(
            list_array64_rpad_and_clip_length_axis1(&[0, 4], &[4, 8], 2),
            8
        );
    }
}
