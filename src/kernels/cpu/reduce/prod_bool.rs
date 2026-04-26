// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: boolean AND (all non-zero) per group.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_prod_bool.cpp`.

/// For each group, `toptr[group] = all elements != 0`.
/// `toptr` is initialised to `true`.
pub fn reduce_prod_bool<IN>(toptr: &mut [bool], fromptr: &[IN], parents: &[i64])
where
    IN: PartialEq + Default + Copy,
{
    assert_eq!(fromptr.len(), parents.len());
    for v in toptr.iter_mut() {
        *v = true;
    }
    let zero = IN::default();
    for (&val, &p) in fromptr.iter().zip(parents.iter()) {
        if val == zero {
            toptr[p as usize] = false;
        }
    }
}

macro_rules! impl_prod_bool {
    ($fn_name:ident, $t:ty) => {
        pub fn $fn_name(toptr: &mut [bool], fromptr: &[$t], parents: &[i64]) {
            reduce_prod_bool(toptr, fromptr, parents)
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

pub fn reduce_prod_bool_float32_64(toptr: &mut [bool], fromptr: &[f32], parents: &[i64]) {
    assert_eq!(fromptr.len(), parents.len());
    for v in toptr.iter_mut() {
        *v = true;
    }
    for (&val, &p) in fromptr.iter().zip(parents.iter()) {
        if val == 0.0_f32 {
            toptr[p as usize] = false;
        }
    }
}

pub fn reduce_prod_bool_float64_64(toptr: &mut [bool], fromptr: &[f64], parents: &[i64]) {
    assert_eq!(fromptr.len(), parents.len());
    for v in toptr.iter_mut() {
        *v = true;
    }
    for (&val, &p) in fromptr.iter().zip(parents.iter()) {
        if val == 0.0_f64 {
            toptr[p as usize] = false;
        }
    }
}

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
