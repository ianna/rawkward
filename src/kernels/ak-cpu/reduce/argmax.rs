// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: index of the maximum element per group.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_argmax.cpp`.

/// For each group, find the index `i` of the maximum value.
/// Groups with no elements retain `-1`.
#[inline]
pub fn reduce_argmax<IN>(toptr: &mut [i64], fromptr: &[IN], parents: &[i64])
where
    IN: PartialOrd + Copy + Default,
{
    assert_eq!(fromptr.len(), parents.len());
    toptr.fill(-1);
    let mut best_val: Vec<IN> = vec![IN::default(); toptr.len()];
    for (i, (&val, &p)) in fromptr.iter().zip(parents.iter()).enumerate() {
        let p = p as usize;
        if toptr[p] == -1 || val > best_val[p] {
            best_val[p] = val;
            toptr[p] = i as i64;
        }
    }
}

macro_rules! impl_argmax {
    ($fn_name:ident, $t:ty) => {
        #[inline]
        pub fn $fn_name(toptr: &mut [i64], fromptr: &[$t], parents: &[i64]) {
            reduce_argmax(toptr, fromptr, parents)
        }
    };
}

impl_argmax!(reduce_argmax_int8_64, i8);
impl_argmax!(reduce_argmax_uint8_64, u8);
impl_argmax!(reduce_argmax_int16_64, i16);
impl_argmax!(reduce_argmax_uint16_64, u16);
impl_argmax!(reduce_argmax_int32_64, i32);
impl_argmax!(reduce_argmax_uint32_64, u32);
impl_argmax!(reduce_argmax_int64_64, i64);
impl_argmax!(reduce_argmax_uint64_64, u64);
impl_argmax!(reduce_argmax_float32_64, f32);
impl_argmax!(reduce_argmax_float64_64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let from = [3i32, 1, 5, 2];
        let parents = [0i64, 0, 1, 1];
        let mut out = [0i64; 2];
        reduce_argmax_int32_64(&mut out, &from, &parents);
        assert_eq!(out, [0, 2]); // index of max in each group
    }

    #[test]
    fn empty_group_retains_minus_one() {
        let from = [5i64];
        let parents = [1i64]; // only group 1 has elements
        let mut out = [0i64; 2];
        reduce_argmax_int64_64(&mut out, &from, &parents);
        assert_eq!(out[0], -1);
        assert_eq!(out[1], 0);
    }
}
