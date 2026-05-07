// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: multiply elements into parent groups.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_prod.cpp`. Offsets-based
//! iteration; see `reduce_sum` for the pattern.

use std::ops::MulAssign;

/// Multiply elements of `fromptr` into `toptr`, grouped by `offsets`.
///
/// `toptr[g]` receives the product of `fromptr[offsets[g]..offsets[g+1]]`,
/// or `1` (the multiplicative identity) if the group is empty.
#[inline]
pub fn reduce_prod<OUT, IN>(toptr: &mut [OUT], fromptr: &[IN], offsets: &[i64])
where
    OUT: One + MulAssign + Copy,
    IN: Copy + Into<OUT>,
{
    assert_eq!(offsets.len(), toptr.len() + 1);
    for (g, slot) in toptr.iter_mut().enumerate() {
        let start = offsets[g] as usize;
        let stop = offsets[g + 1] as usize;
        let mut acc: OUT = OUT::one();
        for &val in &fromptr[start..stop] {
            acc *= val.into();
        }
        *slot = acc;
    }
}

/// Helper trait providing the multiplicative identity `1`.
///
/// This replaces `num_traits::One` to avoid an external dependency.
pub trait One {
    fn one() -> Self;
}

macro_rules! impl_one {
    ($($t:ty, $v:expr);* $(;)?) => {
        $(impl One for $t { fn one() -> Self { $v } })*
    };
}

impl_one! {
    i8,  1;
    u8,  1;
    i16, 1;
    u16, 1;
    i32, 1;
    u32, 1;
    i64, 1;
    u64, 1;
    f32, 1.0;
    f64, 1.0;
}

// ── Typed specialisation wrappers ────────────────────────────────────────────

macro_rules! impl_reduce_prod {
    ($fn_name:ident, $out:ty, $in:ty) => {
        #[doc = concat!(
                    "Product of `", stringify!($in), "` values into `", stringify!($out),
                    "` accumulators per group."
                )]
        #[inline]
        pub fn $fn_name(toptr: &mut [$out], fromptr: &[$in], offsets: &[i64]) {
            reduce_prod(toptr, fromptr, offsets)
        }
    };
}

impl_reduce_prod!(reduce_prod_int64_int8_64, i64, i8);
impl_reduce_prod!(reduce_prod_uint64_uint8_64, u64, u8);
impl_reduce_prod!(reduce_prod_int64_int16_64, i64, i16);
impl_reduce_prod!(reduce_prod_uint64_uint16_64, u64, u16);
impl_reduce_prod!(reduce_prod_int64_int32_64, i64, i32);
impl_reduce_prod!(reduce_prod_uint64_uint32_64, u64, u32);
impl_reduce_prod!(reduce_prod_int64_int64_64, i64, i64);
impl_reduce_prod!(reduce_prod_uint64_uint64_64, u64, u64);
impl_reduce_prod!(reduce_prod_float32_float32_64, f32, f32);
impl_reduce_prod!(reduce_prod_float64_float64_64, f64, f64);
impl_reduce_prod!(reduce_prod_int32_int8_64, i32, i8);
impl_reduce_prod!(reduce_prod_uint32_uint8_64, u32, u8);
impl_reduce_prod!(reduce_prod_int32_int16_64, i32, i16);
impl_reduce_prod!(reduce_prod_uint32_uint16_64, u32, u16);
impl_reduce_prod!(reduce_prod_int32_int32_64, i32, i32);
impl_reduce_prod!(reduce_prod_uint32_uint32_64, u32, u32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_i32() {
        let from = [2i32, 3, 4, 5];
        let offsets = [0i64, 2, 4];
        let mut out = [0i64; 2];
        reduce_prod_int64_int32_64(&mut out, &from, &offsets);
        assert_eq!(out, [6, 20]);
    }

    #[test]
    fn identity_when_empty_group() {
        let from = [3i64];
        let offsets = [0i64, 1, 1]; // group 1 is empty
        let mut out = [0i64; 2];
        reduce_prod_int64_int64_64(&mut out, &from, &offsets);
        assert_eq!(out, [3, 1]);
    }

    #[test]
    fn float_product() {
        let from = [2.0f64, 0.5, 4.0];
        let offsets = [0i64, 2, 3];
        let mut out = [0.0f64; 2];
        reduce_prod_float64_float64_64(&mut out, &from, &offsets);
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn product_includes_zero() {
        let from = [5i32, 0];
        let offsets = [0i64, 2];
        let mut out = [0i64; 1];
        reduce_prod_int64_int32_64(&mut out, &from, &offsets);
        assert_eq!(out[0], 0);
    }

    #[test]
    fn output_initialised_to_one() {
        let from: [i32; 0] = [];
        let offsets = [0i64, 0, 0];
        let mut out = [42i64; 2];
        reduce_prod_int64_int32_64(&mut out, &from, &offsets);
        assert_eq!(out, [1, 1]);
    }
}
