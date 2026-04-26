// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build starts/stops for rpad-and-clip on axis=1.
//!
//! Corresponds to `src/cpu-kernels/awkward_index_rpad_and_clip_axis1.cpp`.

/// For each list `i` in `0..length`, assign a contiguous window of size `target`:
/// * `tostarts[i] = i * target`
/// * `tostops[i]  = (i + 1) * target`
///
/// # Examples
///
/// ```
/// use cpu_kernels::index_rpad_and_clip_axis1::index_rpad_and_clip_axis1_64;
///
/// let mut starts = [0i64; 3];
/// let mut stops  = [0i64; 3];
/// index_rpad_and_clip_axis1_64(&mut starts, &mut stops, 4, 3);
/// assert_eq!(starts, [0, 4, 8]);
/// assert_eq!(stops,  [4, 8, 12]);
/// ```
pub fn index_rpad_and_clip_axis1(tostarts: &mut [i64], tostops: &mut [i64], target: i64, length: usize) {
    assert_eq!(tostarts.len(), length);
    assert_eq!(tostops.len(), length);
    let mut offset = 0i64;
    for i in 0..length {
        tostarts[i] = offset;
        offset += target;
        tostops[i] = offset;
    }
}

/// Typed alias (mirrors `awkward_index_rpad_and_clip_axis1_64`).
pub fn index_rpad_and_clip_axis1_64(tostarts: &mut [i64], tostops: &mut [i64], target: i64, length: usize) {
    index_rpad_and_clip_axis1(tostarts, tostops, target, length);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut starts = [0i64; 3];
        let mut stops  = [0i64; 3];
        index_rpad_and_clip_axis1_64(&mut starts, &mut stops, 4, 3);
        assert_eq!(starts, [0, 4, 8]);
        assert_eq!(stops,  [4, 8, 12]);
    }

    #[test]
    fn target_one() {
        let mut starts = [0i64; 4];
        let mut stops  = [0i64; 4];
        index_rpad_and_clip_axis1_64(&mut starts, &mut stops, 1, 4);
        assert_eq!(starts, [0, 1, 2, 3]);
        assert_eq!(stops,  [1, 2, 3, 4]);
    }

    #[test]
    fn zero_length() {
        let mut starts: [i64; 0] = [];
        let mut stops:  [i64; 0] = [];
        index_rpad_and_clip_axis1_64(&mut starts, &mut stops, 5, 0);
    }
}
