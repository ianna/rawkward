// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Rpad-and-clip each row of a RegularArray to a target width.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_RegularArray_rpad_and_clip_axis1.cpp`.

/// For each row `i`, write `target` entries into `toindex[i*target..(i+1)*target]`:
/// * `j` in `0..shorter`: `toindex[i*target + j] = i*size + j`
/// * `j` in `shorter..target`: `toindex[i*target + j] = -1`
///
/// where `shorter = min(target, size)`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::regular_array_rpad_and_clip_axis1::regular_array_rpad_and_clip_axis1_64;
///
/// // 2 rows of size=2, target=3 → pad with one null per row
/// let mut toindex = [0i64; 6];
/// regular_array_rpad_and_clip_axis1_64(&mut toindex, 3, 2, 2);
/// assert_eq!(toindex, [0, 1, -1, 2, 3, -1]);
/// ```
pub fn regular_array_rpad_and_clip_axis1_64(
    toindex: &mut [i64],
    target: usize,
    size: usize,
    length: usize,
) {
    assert!(toindex.len() >= length * target);
    let shorter = target.min(size);
    for i in 0..length {
        for j in 0..shorter {
            toindex[i * target + j] = (i * size + j) as i64;
        }
        for j in shorter..target {
            toindex[i * target + j] = -1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_short_rows() {
        let mut toindex = [0i64; 6];
        regular_array_rpad_and_clip_axis1_64(&mut toindex, 3, 2, 2);
        assert_eq!(toindex, [0, 1, -1, 2, 3, -1]);
    }

    #[test]
    fn clip_long_rows() {
        // size=5, target=3 → take first 3 per row
        let mut toindex = [0i64; 6];
        regular_array_rpad_and_clip_axis1_64(&mut toindex, 3, 5, 2);
        assert_eq!(toindex, [0, 1, 2, 5, 6, 7]);
    }

    #[test]
    fn exact_fit() {
        let mut toindex = [0i64; 6];
        regular_array_rpad_and_clip_axis1_64(&mut toindex, 3, 3, 2);
        assert_eq!(toindex, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn all_padding() {
        // size=0, target=2 → all -1
        let mut toindex = [0i64; 4];
        regular_array_rpad_and_clip_axis1_64(&mut toindex, 2, 0, 2);
        assert_eq!(toindex, [-1, -1, -1, -1]);
    }

    #[test]
    fn zero_length() {
        let mut toindex: [i64; 0] = [];
        regular_array_rpad_and_clip_axis1_64(&mut toindex, 3, 2, 0);
    }
}
