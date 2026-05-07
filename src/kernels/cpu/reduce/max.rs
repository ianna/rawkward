// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: maximum element per group.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_max.cpp`. Offsets-based
//! iteration; see `reduce_sum` for the pattern.

use std::cmp::PartialOrd;

/// Compute the maximum element in each parent group.
///
/// `toptr[g]` receives the maximum of `fromptr[offsets[g]..offsets[g+1]]`,
/// or `identity` if the group is empty.
#[inline]
pub fn reduce_max<OUT, IN>(toptr: &mut [OUT], fromptr: &[IN], offsets: &[i64], identity: OUT)
where
    OUT: PartialOrd + Copy,
    IN: Copy + Into<OUT>,
{
    assert_eq!(offsets.len(), toptr.len() + 1);
    for (g, slot) in toptr.iter_mut().enumerate() {
        let start = offsets[g] as usize;
        let stop = offsets[g + 1] as usize;
        let mut best: OUT = identity;
        for &val in &fromptr[start..stop] {
            let x: OUT = val.into();
            if x > best {
                best = x;
            }
        }
        *slot = best;
    }
}

// ── Typed specialisation wrappers ────────────────────────────────────────────

macro_rules! impl_reduce_max {
    ($fn_name:ident, $t:ty) => {
        #[doc = concat!(
                    "Maximum of `", stringify!($t), "` values per group."
                )]
        #[inline]
        pub fn $fn_name(toptr: &mut [$t], fromptr: &[$t], offsets: &[i64], identity: $t) {
            reduce_max(toptr, fromptr, offsets, identity)
        }
    };
}

impl_reduce_max!(reduce_max_int8_int8_64, i8);
impl_reduce_max!(reduce_max_uint8_uint8_64, u8);
impl_reduce_max!(reduce_max_int16_int16_64, i16);
impl_reduce_max!(reduce_max_uint16_uint16_64, u16);
impl_reduce_max!(reduce_max_int32_int32_64, i32);
impl_reduce_max!(reduce_max_uint32_uint32_64, u32);
impl_reduce_max!(reduce_max_int64_int64_64, i64);
impl_reduce_max!(reduce_max_uint64_uint64_64, u64);
impl_reduce_max!(reduce_max_float32_float32_64, f32);
impl_reduce_max!(reduce_max_float64_float64_64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_i64() {
        let from = [3i64, 1, 5, 2];
        let offsets = [0i64, 2, 4];
        let mut out = [0i64; 2];
        reduce_max_int64_int64_64(&mut out, &from, &offsets, i64::MIN);
        assert_eq!(out, [3, 5]);
    }

    #[test]
    fn identity_when_no_elements() {
        let from: [i32; 0] = [];
        let offsets = [0i64, 0, 0];
        let mut out = [0i32; 2];
        reduce_max_int32_int32_64(&mut out, &from, &offsets, i32::MIN);
        assert_eq!(out, [i32::MIN, i32::MIN]);
    }

    #[test]
    fn single_element_groups() {
        let from = [7i32, 3];
        let offsets = [0i64, 1, 2];
        let mut out = [0i32; 2];
        reduce_max_int32_int32_64(&mut out, &from, &offsets, i32::MIN);
        assert_eq!(out, [7, 3]);
    }

    #[test]
    fn unsigned_max() {
        let from = [10u8, 200, 50];
        let offsets = [0i64, 2, 3];
        let mut out = [0u8; 2];
        reduce_max_uint8_uint8_64(&mut out, &from, &offsets, u8::MIN);
        assert_eq!(out, [200, 50]);
    }

    #[test]
    fn float_max() {
        let from = [1.5f64, -2.0, 3.0, 0.5];
        let offsets = [0i64, 2, 4];
        let mut out = [0.0f64; 2];
        reduce_max_float64_float64_64(&mut out, &from, &offsets, f64::NEG_INFINITY);
        assert!((out[0] - 1.5).abs() < 1e-12);
        assert!((out[1] - 3.0).abs() < 1e-12);
    }
}
