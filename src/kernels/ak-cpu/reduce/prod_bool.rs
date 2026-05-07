// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: boolean AND (all non-zero) per group.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_prod_bool.cpp`.

/// For each group, `toptr[group] = all elements != 0`.
/// `toptr` is initialised to `true`.
#[inline]
pub fn reduce_prod_bool<IN>(toptr: &mut [bool], fromptr: &[IN], parents: &[i64])
where
    IN: PartialEq + Default + Copy,
{
    assert_eq!(fromptr.len(), parents.len());
    toptr.fill(true);
    let zero = IN::default();
    for (&val, &p) in fromptr.iter().zip(parents.iter()) {
        if val == zero {
            toptr[p as usize] = false;
        }
    }
}

macro_rules! impl_prod_bool {
    ($fn_name:ident, $t:ty) => {
        #[inline]
        pub fn $fn_name(toptr: &mut [bool], fromptr: &[$t], parents: &[i64]) {
            reduce_prod_bool(toptr, fromptr, parents)
        }
    };
}

// Floats use the generic; `Default::default()` gives `0.0`, and `NaN == 0.0`
// is `false` so NaN never triggers the false-set branch — matching C++.
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
        let parents = [0i64, 0, 1, 1];
        let mut out = [false; 2];
        reduce_prod_bool_int32_64(&mut out, &from, &parents);
        assert_eq!(out, [true, true]);
    }

    #[test]
    fn zero_makes_false() {
        let from = [1i64, 0, 2];
        let parents = [0i64, 0, 1];
        let mut out = [false; 2];
        reduce_prod_bool_int64_64(&mut out, &from, &parents);
        assert_eq!(out, [false, true]);
    }

    #[test]
    fn empty_group_stays_true() {
        // group 1 gets no elements → true (vacuous truth)
        let from = [1i32];
        let parents = [0i64];
        let mut out = [false; 2];
        reduce_prod_bool_int32_64(&mut out, &from, &parents);
        assert_eq!(out, [true, true]);
    }
}
