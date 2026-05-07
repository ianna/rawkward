// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: index of the minimum element per group.
//!
<<<<<<< HEAD
//! Corresponds to `src/cpu-kernels/awkward_reduce_argmin.cpp`. Offsets-
//! based iteration; see `reduce_argmax` for the pattern.

/// For each group, find the global flat index of the minimum value in
/// `fromptr[offsets[g]..offsets[g+1]]`. Empty groups produce `-1`.
#[inline]
pub fn reduce_argmin<IN>(toptr: &mut [i64], fromptr: &[IN], offsets: &[i64])
where
    IN: PartialOrd + Copy,
{
    assert_eq!(offsets.len(), toptr.len() + 1);
    for (g, slot) in toptr.iter_mut().enumerate() {
        let start = offsets[g] as usize;
        let stop = offsets[g + 1] as usize;
        let mut best_idx: i64 = -1;
        let mut best_val: Option<IN> = None;
        for (i, &val) in fromptr[start..stop].iter().enumerate() {
            let global = (start + i) as i64;
            match best_val {
                None => {
                    best_val = Some(val);
                    best_idx = global;
                }
                Some(b) if val < b => {
                    best_val = Some(val);
                    best_idx = global;
                }
                _ => {}
            }
        }
        *slot = best_idx;
=======
//! Corresponds to `src/cpu-kernels/awkward_reduce_argmin.cpp`.

/// For each group, find the index `i` (global flat index) of the minimum value
/// in `fromptr[i]` among all `i` in that group.
///
/// `toptr` is initialised to `-1` (no value seen yet). A group with no elements
/// retains `-1`.
pub fn reduce_argmin<IN>(toptr: &mut [i64], fromptr: &[IN], parents: &[i64])
where
    IN: PartialOrd + Copy,
{
    assert_eq!(fromptr.len(), parents.len());
    for v in toptr.iter_mut() {
        *v = -1;
    }
    for (i, (&val, &p)) in fromptr.iter().zip(parents.iter()).enumerate() {
        let best = toptr[p as usize];
        if best == -1 || val < fromptr[best as usize] {
            toptr[p as usize] = i as i64;
        }
>>>>>>> origin/main
    }
}

macro_rules! impl_argmin {
    ($fn_name:ident, $t:ty) => {
<<<<<<< HEAD
        #[inline]
        pub fn $fn_name(toptr: &mut [i64], fromptr: &[$t], offsets: &[i64]) {
            reduce_argmin(toptr, fromptr, offsets)
=======
        pub fn $fn_name(toptr: &mut [i64], fromptr: &[$t], parents: &[i64]) {
            reduce_argmin(toptr, fromptr, parents)
>>>>>>> origin/main
        }
    };
}

impl_argmin!(reduce_argmin_int8_64, i8);
impl_argmin!(reduce_argmin_uint8_64, u8);
impl_argmin!(reduce_argmin_int16_64, i16);
impl_argmin!(reduce_argmin_uint16_64, u16);
impl_argmin!(reduce_argmin_int32_64, i32);
impl_argmin!(reduce_argmin_uint32_64, u32);
impl_argmin!(reduce_argmin_int64_64, i64);
impl_argmin!(reduce_argmin_uint64_64, u64);
impl_argmin!(reduce_argmin_float32_64, f32);
impl_argmin!(reduce_argmin_float64_64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let from = [3i32, 1, 5, 2];
<<<<<<< HEAD
        let offsets = [0i64, 2, 4];
        let mut out = [0i64; 2];
        reduce_argmin_int32_64(&mut out, &from, &offsets);
        assert_eq!(out, [1, 3]);
    }

    #[test]
    fn empty_group_yields_minus_one() {
        let from = [5i64];
        let offsets = [0i64, 1, 1]; // group 1 empty
        let mut out = [0i64; 2];
        reduce_argmin_int64_64(&mut out, &from, &offsets);
=======
        let parents = [0i64, 0, 1, 1];
        let mut out = [0i64; 2];
        reduce_argmin_int32_64(&mut out, &from, &parents);
        assert_eq!(out, [1, 3]); // index of min in each group
    }

    #[test]
    fn empty_group_retains_minus_one() {
        let from = [5i64];
        let parents = [0i64];
        let mut out = [0i64; 2];
        reduce_argmin_int64_64(&mut out, &from, &parents);
>>>>>>> origin/main
        assert_eq!(out[0], 0);
        assert_eq!(out[1], -1);
    }

    #[test]
    fn float_argmin() {
        let from = [2.0f64, -1.0, 3.0, 0.5];
<<<<<<< HEAD
        let offsets = [0i64, 2, 4];
        let mut out = [0i64; 2];
        reduce_argmin_float64_64(&mut out, &from, &offsets);
=======
        let parents = [0i64, 0, 1, 1];
        let mut out = [0i64; 2];
        reduce_argmin_float64_64(&mut out, &from, &parents);
>>>>>>> origin/main
        assert_eq!(out, [1, 3]);
    }
}
