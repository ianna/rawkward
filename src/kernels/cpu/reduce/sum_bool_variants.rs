// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Count-style sum of bool inputs into integer accumulators.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_sum_int64_bool_64.cpp` and
//! `src/cpu-kernels/awkward_reduce_sum_int32_bool_64.cpp`. Offsets-based
//! iteration; see `reduce_sum` for the pattern.

/// Sum `bool` inputs (treating `true` as `1`) into an `i64` accumulator per group.
#[inline]
pub fn reduce_sum_int64_bool_64(toptr: &mut [i64], fromptr: &[bool], offsets: &[i64]) {
    assert_eq!(offsets.len(), toptr.len() + 1);
    for (g, slot) in toptr.iter_mut().enumerate() {
        let start = offsets[g] as usize;
        let stop = offsets[g + 1] as usize;
        let mut acc: i64 = 0;
        for &val in &fromptr[start..stop] {
            acc += val as i64;
        }
        *slot = acc;
    }
}

/// Sum `bool` inputs into an `i32` accumulator per group.
#[inline]
pub fn reduce_sum_int32_bool_64(toptr: &mut [i32], fromptr: &[bool], offsets: &[i64]) {
    assert_eq!(offsets.len(), toptr.len() + 1);
    for (g, slot) in toptr.iter_mut().enumerate() {
        let start = offsets[g] as usize;
        let stop = offsets[g + 1] as usize;
        let mut acc: i32 = 0;
        for &val in &fromptr[start..stop] {
            acc += val as i32;
        }
        *slot = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i64_count() {
        let from = [true, false, true, true];
        let offsets = [0i64, 2, 4];
        let mut out = [0i64; 2];
        reduce_sum_int64_bool_64(&mut out, &from, &offsets);
        assert_eq!(out, [1, 2]);
    }

    #[test]
    fn i32_count() {
        let from = [true, true, false];
        let offsets = [0i64, 3];
        let mut out = [0i32; 1];
        reduce_sum_int32_bool_64(&mut out, &from, &offsets);
        assert_eq!(out[0], 2);
    }
}
