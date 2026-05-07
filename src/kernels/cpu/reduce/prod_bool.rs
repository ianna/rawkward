// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: boolean AND (all non-zero) per group.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_prod_bool.cpp`. Offsets-
//! based iteration with per-group early-exit on the first zero element.

/// For each group, set `toptr[g] = true` iff all
/// `fromptr[offsets[g]..offsets[g+1]] != 0`. Empty groups are vacuously true.
#[inline]
pub fn reduce_prod_bool<IN>(toptr: &mut [bool], fromptr: &[IN], offsets: &[i64])
where
    IN: PartialEq + Default + Copy,
{
    assert_eq!(offsets.len(), toptr.len() + 1);
    let zero = IN::default();
    for (g, slot) in toptr.iter_mut().enumerate() {
        let start = offsets[g] as usize;
        let stop = offsets[g + 1] as usize;
        let mut all = true;
        for &val in &fromptr[start..stop] {
            if val == zero {
                all = false;
                break;
            }
        }
        *slot = all;
    }
}

macro_rules! impl_prod_bool {
    ($fn_name:ident, $t:ty) => {
        #[inline]
        pub fn $fn_name(toptr: &mut [bool], fromptr: &[$t], offsets: &[i64]) {
            reduce_prod_bool(toptr, fromptr, offsets)
        }
    };
}

impl_prod_bool!(reduce_prod_bool_bool_64, bool);
impl_prod_bool!(reduce_prod_bool_int8_64, i8);
impl_prod_bool!(reduce_prod_bool_uint8_64, u8);
impl_prod_bool!(reduce_prod_bool_int16_64, i16);
impl_prod_bool!(reduce_prod_bool_uint16_64, u16);
impl_prod_bool!(reduce_prod_bool_int32_64, i32);
impl_prod_bool!(reduce_prod_bool_uint32_64, u32);
impl_prod_bool!(reduce_prod_bool_int64_64, i64);
impl_prod_bool!(reduce_prod_bool_uint64_64, u64);
impl_prod_bool!(reduce_prod_bool_float32_64, f32);
impl_prod_bool!(reduce_prod_bool_float64_64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_nonzero() {
        let from = [1i32, 2, 3, 4];
        let offsets = [0i64, 2, 4];
        let mut out = [false; 2];
        reduce_prod_bool_int32_64(&mut out, &from, &offsets);
        assert_eq!(out, [true, true]);
    }

    #[test]
    fn zero_makes_false() {
        let from = [1i64, 0, 2];
        let offsets = [0i64, 2, 3];
        let mut out = [false; 2];
        reduce_prod_bool_int64_64(&mut out, &from, &offsets);
        assert_eq!(out, [false, true]);
    }

    #[test]
    fn empty_group_stays_true() {
        let from = [1i32];
        let offsets = [0i64, 1, 1];
        let mut out = [false; 2];
        reduce_prod_bool_int32_64(&mut out, &from, &offsets);
        assert_eq!(out, [true, true]);
    }
}
