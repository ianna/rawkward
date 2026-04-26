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
pub fn missing_repeat(
    outindex: &mut [i64],
    index: &[i64],
    repetitions: i64,
    regularsize: i64,
) {
    let indexlength = index.len() as i64;
    for i in 0..repetitions {
        let out_offset = (i * indexlength) as usize;
        let val_offset = i * regularsize;
        for j in 0..indexlength as usize {
            let base = index[j];
            let adjustment = if base >= 0 { val_offset } else { 0 };
            outindex[out_offset + j] = base + adjustment;
        }
    }
}

/// Typed alias (mirrors `awkward_missing_repeat_64`).
pub fn missing_repeat_64(
    outindex: &mut [i64],
    index: &[i64],
    repetitions: i64,
    regularsize: i64,
) {
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
