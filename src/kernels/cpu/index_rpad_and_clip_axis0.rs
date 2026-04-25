// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build the index for rpad-and-clip on axis=0.
//!
//! Corresponds to `src/cpu-kernels/awkward_index_rpad_and_clip_axis0.cpp`.

/// Fill `toindex[0..shorter]` with `0, 1, …, shorter-1` and
/// `toindex[shorter..target]` with `-1`, where `shorter = min(target, length)`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::index_rpad_and_clip_axis0::index_rpad_and_clip_axis0_64;
///
/// let mut idx = [0i64; 5];
/// index_rpad_and_clip_axis0_64(&mut idx, 5, 3); // target=5, length=3
/// assert_eq!(idx, [0, 1, 2, -1, -1]);
/// ```
pub fn index_rpad_and_clip_axis0(toindex: &mut [i64], target: usize, length: usize) {
    let shorter = target.min(length);
    for i in 0..shorter {
        toindex[i] = i as i64;
    }
    for i in shorter..target {
        toindex[i] = -1;
    }
}

/// Typed alias (mirrors `awkward_index_rpad_and_clip_axis0_64`).
pub fn index_rpad_and_clip_axis0_64(toindex: &mut [i64], target: usize, length: usize) {
    index_rpad_and_clip_axis0(toindex, target, length);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pads_with_minus_one() {
        let mut idx = [0i64; 5];
        index_rpad_and_clip_axis0_64(&mut idx, 5, 3);
        assert_eq!(idx, [0, 1, 2, -1, -1]);
    }

    #[test]
    fn clips_to_target() {
        let mut idx = [0i64; 3];
        index_rpad_and_clip_axis0_64(&mut idx, 3, 10);
        assert_eq!(idx, [0, 1, 2]);
    }

    #[test]
    fn exact_fit() {
        let mut idx = [0i64; 4];
        index_rpad_and_clip_axis0_64(&mut idx, 4, 4);
        assert_eq!(idx, [0, 1, 2, 3]);
    }

    #[test]
    fn zero_length() {
        let mut idx = [0i64; 3];
        index_rpad_and_clip_axis0_64(&mut idx, 3, 0);
        assert_eq!(idx, [-1, -1, -1]);
    }
}
