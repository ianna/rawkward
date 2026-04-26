// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: boolean OR (any non-zero) per group.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_sum_bool.cpp`.

/// For each group, set `toptr[group] = true` if any `fromptr[i] != 0` in that group.
///
/// `toptr` is zero-initialised before the loop.
#[inline]
pub fn reduce_sum_bool<IN>(toptr: &mut [bool], fromptr: &[IN], parents: &[i64])
where
    IN: PartialEq + Default + Copy,
{
    assert_eq!(fromptr.len(), parents.len());
    toptr.fill(false);
    let zero = IN::default();
    for (&val, &p) in fromptr.iter().zip(parents.iter()) {
        if val != zero {
            toptr[p as usize] = true;
        }
    }
}

macro_rules! impl_sum_bool {
    ($fn_name:ident, $t:ty) => {
        #[inline]
        pub fn $fn_name(toptr: &mut [bool], fromptr: &[$t], parents: &[i64]) {
            reduce_sum_bool(toptr, fromptr, parents)
        }
    };
}

// Floats use the generic too: `Default::default()` gives `0.0` and
// `NaN != 0.0` is `true`, matching the C++ `fromptr[i] != 0` semantics.
impl_sum_bool!(reduce_sum_bool_bool_64, bool);
impl_sum_bool!(reduce_sum_bool_int8_64, i8);
impl_sum_bool!(reduce_sum_bool_uint8_64, u8);
impl_sum_bool!(reduce_sum_bool_int16_64, i16);
impl_sum_bool!(reduce_sum_bool_uint16_64, u16);
impl_sum_bool!(reduce_sum_bool_int32_64, i32);
impl_sum_bool!(reduce_sum_bool_uint32_64, u32);
impl_sum_bool!(reduce_sum_bool_int64_64, i64);
impl_sum_bool!(reduce_sum_bool_uint64_64, u64);
impl_sum_bool!(reduce_sum_bool_float32_64, f32);
impl_sum_bool!(reduce_sum_bool_float64_64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_nonzero() {
        let from = [0i32, 1, 0, 0];
        let parents = [0i64, 0, 1, 1];
        let mut out = [false; 2];
        reduce_sum_bool_int32_64(&mut out, &from, &parents);
        assert_eq!(out, [true, false]);
    }

    #[test]
    fn all_zero() {
        let from = [0i64; 4];
        let parents = [0i64, 0, 1, 1];
        let mut out = [false; 2];
        reduce_sum_bool_int64_64(&mut out, &from, &parents);
        assert_eq!(out, [false, false]);
    }

    #[test]
    fn bool_input() {
        let from = [false, true, false];
        let parents = [0i64, 0, 0];
        let mut out = [false; 1];
        reduce_sum_bool_bool_64(&mut out, &from, &parents);
        assert!(out[0]);
    }
}
