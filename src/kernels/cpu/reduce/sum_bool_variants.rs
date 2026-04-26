// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Count-style sum of bool inputs into integer accumulators.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_sum_int64_bool_64.cpp` and
//! `src/cpu-kernels/awkward_reduce_sum_int32_bool_64.cpp`.

/// Sum `bool` inputs (treating `true` as `1`) into an `i64` accumulator per group.
pub fn reduce_sum_int64_bool_64(toptr: &mut [i64], fromptr: &[bool], parents: &[i64]) {
    assert_eq!(fromptr.len(), parents.len());
    for v in toptr.iter_mut() {
        *v = 0;
    }
    for (&val, &p) in fromptr.iter().zip(parents.iter()) {
        toptr[p as usize] += val as i64;
    }
}

/// Sum `bool` inputs into an `i32` accumulator per group.
pub fn reduce_sum_int32_bool_64(toptr: &mut [i32], fromptr: &[bool], parents: &[i64]) {
    assert_eq!(fromptr.len(), parents.len());
    for v in toptr.iter_mut() {
        *v = 0;
    }
    for (&val, &p) in fromptr.iter().zip(parents.iter()) {
        toptr[p as usize] += val as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i64_count() {
        let from = [true, false, true, true];
        let parents = [0i64, 0, 1, 1];
        let mut out = [0i64; 2];
        reduce_sum_int64_bool_64(&mut out, &from, &parents);
        assert_eq!(out, [1, 2]);
    }

    #[test]
    fn i32_count() {
        let from = [true, true, false];
        let parents = [0i64, 0, 0];
        let mut out = [0i32; 1];
        reduce_sum_int32_bool_64(&mut out, &from, &parents);
        assert_eq!(out[0], 2);
    }
}
