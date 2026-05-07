// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: sum elements into parent groups.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_sum.cpp`.
//!
//! Iteration is **offsets-based**: for each output group `g`, accumulate
//! `fromptr[offsets[g]..offsets[g+1]]` into a register-resident `acc` and
//! store it once. This matches awkward C++ and lets LLVM auto-vectorize
//! the inner loop. The previous parents-based scatter (`toptr[parents[i]]
//! += val`) defeated the autovectorizer because of the indirect store.

use std::ops::AddAssign;

/// Sum elements of `fromptr` into `toptr`, grouped by `offsets`.
///
/// `toptr[g]` receives the sum of `fromptr[offsets[g]..offsets[g+1]]`.
/// The output length is `toptr.len()` and `offsets.len()` must equal
/// `toptr.len() + 1`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::reduce_sum::reduce_sum_int64_int8_64;
///
/// let from    = [1i8, 2, 3, 4];
/// let offsets = [0i64, 2, 4]; // two groups of two
/// let mut out = [0i64; 2];
/// reduce_sum_int64_int8_64(&mut out, &from, &offsets);
/// assert_eq!(out, [3, 7]);
/// ```
#[inline]
pub fn reduce_sum<OUT, IN>(toptr: &mut [OUT], fromptr: &[IN], offsets: &[i64])
where
    OUT: Default + AddAssign + Copy,
    IN: Copy + Into<OUT>,
{
    assert_eq!(offsets.len(), toptr.len() + 1);
    for (g, slot) in toptr.iter_mut().enumerate() {
        let start = offsets[g] as usize;
        let stop = offsets[g + 1] as usize;
        let mut acc: OUT = OUT::default();
        for &val in &fromptr[start..stop] {
            acc += val.into();
        }
        *slot = acc;
    }
}

// ── Typed specialisation wrappers ────────────────────────────────────────────

macro_rules! impl_reduce_sum {
    ($fn_name:ident, $out:ty, $in:ty) => {
        #[doc = concat!(
                            "Sum `", stringify!($in), "` values into `", stringify!($out),
                            "` accumulators per group."
                        )]
        #[inline]
        pub fn $fn_name(toptr: &mut [$out], fromptr: &[$in], offsets: &[i64]) {
            reduce_sum(toptr, fromptr, offsets)
        }
    };
}

impl_reduce_sum!(reduce_sum_int64_int8_64, i64, i8);
impl_reduce_sum!(reduce_sum_uint64_uint8_64, u64, u8);
impl_reduce_sum!(reduce_sum_int64_int16_64, i64, i16);
impl_reduce_sum!(reduce_sum_uint64_uint16_64, u64, u16);
impl_reduce_sum!(reduce_sum_int64_int32_64, i64, i32);
impl_reduce_sum!(reduce_sum_uint64_uint32_64, u64, u32);
impl_reduce_sum!(reduce_sum_int64_int64_64, i64, i64);
impl_reduce_sum!(reduce_sum_uint64_uint64_64, u64, u64);
impl_reduce_sum!(reduce_sum_float32_float32_64, f32, f32);
impl_reduce_sum!(reduce_sum_float64_float64_64, f64, f64);
impl_reduce_sum!(reduce_sum_int32_int8_64, i32, i8);
impl_reduce_sum!(reduce_sum_uint32_uint8_64, u32, u8);
impl_reduce_sum!(reduce_sum_int32_int16_64, i32, i16);
impl_reduce_sum!(reduce_sum_uint32_uint16_64, u32, u16);
impl_reduce_sum!(reduce_sum_int32_int32_64, i32, i32);
impl_reduce_sum!(reduce_sum_uint32_uint32_64, u32, u32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_i8_to_i64() {
        let from = [1i8, 2, 3, 4];
        let offsets = [0i64, 2, 4];
        let mut out = [0i64; 2];
        reduce_sum_int64_int8_64(&mut out, &from, &offsets);
        assert_eq!(out, [3, 7]);
    }

    #[test]
    fn float64_sum() {
        let from = [1.5f64, 2.5, 3.0];
        let offsets = [0i64, 2, 3];
        let mut out = [0.0f64; 2];
        reduce_sum_float64_float64_64(&mut out, &from, &offsets);
        assert!((out[0] - 4.0).abs() < 1e-12);
        assert!((out[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn single_group() {
        let from = [10i32, 20, 30];
        let offsets = [0i64, 3];
        let mut out = [0i32; 1];
        reduce_sum_int32_int32_64(&mut out, &from, &offsets);
        assert_eq!(out[0], 60);
    }

    #[test]
    fn output_zeroed_first() {
        let from = [5i64];
        let offsets = [0i64, 1];
        let mut out = [99i64; 1];
        reduce_sum_int64_int64_64(&mut out, &from, &offsets);
        assert_eq!(out[0], 5);
    }

    #[test]
    fn empty_group() {
        let from = [10i64, 20];
        let offsets = [0i64, 2, 2]; // group 1 is empty
        let mut out = [0i64; 2];
        reduce_sum_int64_int64_64(&mut out, &from, &offsets);
        assert_eq!(out, [30, 0]);
    }

    #[test]
    fn unsigned_sum() {
        let from = [100u32, 200, 300];
        let offsets = [0i64, 1, 3];
        let mut out = [0u64; 2];
        reduce_sum_uint64_uint32_64(&mut out, &from, &offsets);
        assert_eq!(out, [100, 500]);
    }
}
