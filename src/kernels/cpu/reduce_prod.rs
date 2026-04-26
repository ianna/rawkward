// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: multiply elements into parent groups.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_prod.cpp`.

use std::ops::MulAssign;

/// Multiply elements from `fromptr` into `toptr`, grouped by `parents`.
///
/// `toptr` is initialised to `1` (the multiplicative identity), then for
/// each `i`: `toptr[parents[i]] *= OUT::from(fromptr[i])`.
///
/// # Type parameters
///
/// * `OUT` – Accumulator type; must support `MulAssign` and be constructable
///   as the value `1`.
/// * `IN`  – Input type; must be convertible into `OUT`.
///
/// # Panics
///
/// Panics if `fromptr.len() != parents.len()` or if any parent index is out
/// of range for `toptr`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::reduce_prod::reduce_prod;
///
/// let from    = [2i32, 3, 4, 5];
/// let parents = [0i64, 0, 1, 1];
/// let mut out = [0i64; 2];
/// reduce_prod(&mut out, &from, &parents);
/// assert_eq!(out, [6, 20]);
/// ```
pub fn reduce_prod<OUT, IN>(toptr: &mut [OUT], fromptr: &[IN], parents: &[i64])
where
    OUT: One + MulAssign + Copy,
    IN: Copy + Into<OUT>,
{
    assert_eq!(fromptr.len(), parents.len());
    for v in toptr.iter_mut() {
        *v = OUT::one();
    }
    for (&val, &p) in fromptr.iter().zip(parents.iter()) {
        toptr[p as usize] *= val.into();
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
        pub fn $fn_name(toptr: &mut [$out], fromptr: &[$in], parents: &[i64]) {
            reduce_prod(toptr, fromptr, parents)
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
        let parents = [0i64, 0, 1, 1];
        let mut out = [0i64; 2];
        reduce_prod_int64_int32_64(&mut out, &from, &parents);
        assert_eq!(out, [6, 20]);
    }

    #[test]
    fn identity_when_empty_group() {
        // Group 1 gets no elements → remains 1
        let from = [3i64];
        let parents = [0i64];
        let mut out = [0i64; 2];
        reduce_prod_int64_int64_64(&mut out, &from, &parents);
        assert_eq!(out, [3, 1]);
    }

    #[test]
    fn float_product() {
        let from = [2.0f64, 0.5, 4.0];
        let parents = [0i64, 0, 1];
        let mut out = [0.0f64; 2];
        reduce_prod_float64_float64_64(&mut out, &from, &parents);
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn product_includes_zero() {
        let from = [5i32, 0];
        let parents = [0i64, 0];
        let mut out = [0i64; 1];
        reduce_prod_int64_int32_64(&mut out, &from, &parents);
        assert_eq!(out[0], 0);
    }

    #[test]
    fn output_initialised_to_one() {
        // No input elements → output should be multiplicative identity
        let from: [i32; 0] = [];
        let parents: [i64; 0] = [];
        let mut out = [42i64; 2]; // should be overwritten with 1, not 42
        reduce_prod_int64_int32_64(&mut out, &from, &parents);
        assert_eq!(out, [1, 1]);
    }
}
