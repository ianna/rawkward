// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build the output index for rpad-and-clip on axis=1 of an IndexedOptionArray.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_IndexedOptionArray_rpad_and_clip_mask_axis1.cpp`.

/// For each position `i`:
/// * `frommask[i] != 0` → `toindex[i] = -1` (masked/null).
/// * `frommask[i] == 0` → `toindex[i] = count`, then `count++`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_option_array_rpad_and_clip_mask_axis1::indexed_option_array_rpad_and_clip_mask_axis1_64;
///
/// let mask = [0i8, 1, 0, 0, 1];
/// let mut idx = [0i64; 5];
/// indexed_option_array_rpad_and_clip_mask_axis1_64(&mut idx, &mask);
/// assert_eq!(idx, [0, -1, 1, 2, -1]);
/// ```
pub fn indexed_option_array_rpad_and_clip_mask_axis1(toindex: &mut [i64], frommask: &[i8]) {
    assert_eq!(toindex.len(), frommask.len());
    let mut count = 0i64;
    for (out, &m) in toindex.iter_mut().zip(frommask.iter()) {
        if m != 0 {
            *out = -1;
        } else {
            *out = count;
            count += 1;
        }
    }
}

/// Typed alias (mirrors `awkward_IndexedOptionArray_rpad_and_clip_mask_axis1_64`).
pub fn indexed_option_array_rpad_and_clip_mask_axis1_64(toindex: &mut [i64], frommask: &[i8]) {
    indexed_option_array_rpad_and_clip_mask_axis1(toindex, frommask);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mask = [0i8, 1, 0, 0, 1];
        let mut idx = [0i64; 5];
        indexed_option_array_rpad_and_clip_mask_axis1_64(&mut idx, &mask);
        assert_eq!(idx, [0, -1, 1, 2, -1]);
    }

    #[test]
    fn all_unmasked() {
        let mask = [0i8; 4];
        let mut idx = [0i64; 4];
        indexed_option_array_rpad_and_clip_mask_axis1_64(&mut idx, &mask);
        assert_eq!(idx, [0, 1, 2, 3]);
    }

    #[test]
    fn all_masked() {
        let mask = [1i8; 4];
        let mut idx = [0i64; 4];
        indexed_option_array_rpad_and_clip_mask_axis1_64(&mut idx, &mask);
        assert_eq!(idx, [-1, -1, -1, -1]);
    }
}
