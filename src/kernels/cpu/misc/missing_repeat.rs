// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Repeat an index array, adjusting non-null values by a regular-size stride.
//!
//! Corresponds to `src/cpu-kernels/awkward_missing_repeat.cpp`.

/// Tile `index` into `outindex` `repetitions` times.
///
/// For repetition `i` (0-based) and position `j` in `index`:
/// * If `index[j] >= 0`: `outindex[i * indexlength + j] = index[j] + i * regularsize`.
/// * If `index[j] < 0`: `outindex[i * indexlength + j] = index[j]` (null stays null).
///
/// # Examples
///
/// ```
/// use cpu_kernels::missing_repeat::missing_repeat_64;
///
/// let index = [0i64, -1, 1]; // indexlength=3, regularsize=2
/// let mut out = [0i64; 6];
/// missing_repeat_64(&mut out, &index, 2, 2);
/// // rep 0: [0+0, -1, 1+0] = [0, -1, 1]
/// // rep 1: [0+2, -1, 1+2] = [2, -1, 3]
/// assert_eq!(out, [0, -1, 1, 2, -1, 3]);
/// ```
#[inline]
pub fn missing_repeat(outindex: &mut [i64], index: &[i64], repetitions: i64, regularsize: i64) {
    let indexlength = index.len();
    for i in 0..repetitions as usize {
        let out_offset = i * indexlength;
        let val_offset = (i as i64) * regularsize;
        // Branchless: a sign-bit mask zeroes the offset when base < 0.
        // `(base >> 63)` is `-1` if negative, `0` otherwise; `& !mask`
        // therefore preserves `val_offset` only when base is non-negative.
        let dst = &mut outindex[out_offset..out_offset + indexlength];
        for (slot, &base) in dst.iter_mut().zip(index.iter()) {
            let mask = base >> 63; // -1 if base < 0, else 0
            *slot = base + (val_offset & !mask);
        }
    }
}

/// Typed alias (mirrors `awkward_missing_repeat_64`).
#[inline]
pub fn missing_repeat_64(outindex: &mut [i64], index: &[i64], repetitions: i64, regularsize: i64) {
    missing_repeat(outindex, index, repetitions, regularsize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_repeat() {
        let index = [0i64, -1, 1];
        let mut out = [0i64; 6];
        missing_repeat_64(&mut out, &index, 2, 2);
        assert_eq!(out, [0, -1, 1, 2, -1, 3]);
    }

    #[test]
    fn single_repetition() {
        let index = [2i64, -1];
        let mut out = [0i64; 2];
        missing_repeat_64(&mut out, &index, 1, 5);
        assert_eq!(out, [2, -1]);
    }

    #[test]
    fn all_null() {
        let index = [-1i64, -1];
        let mut out = [0i64; 6];
        missing_repeat_64(&mut out, &index, 3, 10);
        assert_eq!(out, [-1, -1, -1, -1, -1, -1]);
    }

    #[test]
    fn zero_repetitions() {
        let index = [0i64];
        let mut out: [i64; 0] = [];
        missing_repeat_64(&mut out, &index, 0, 5);
        // No-op — just checking it doesn't panic.
    }
}
