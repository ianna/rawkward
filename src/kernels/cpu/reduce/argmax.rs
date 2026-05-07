// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: index of the maximum element per group.
//!
<<<<<<< HEAD
//! Corresponds to `src/cpu-kernels/awkward_reduce_argmax.cpp`. Offsets-
//! based iteration: walk each group separately with the running best
//! index/value held in registers, store once. No scatter into `toptr`,
//! no indirect re-read of `fromptr`.

/// For each group, find the global flat index of the maximum value in
/// `fromptr[offsets[g]..offsets[g+1]]`. Empty groups produce `-1`.
#[inline]
pub fn reduce_argmax<IN>(toptr: &mut [i64], fromptr: &[IN], offsets: &[i64])
where
    IN: PartialOrd + Copy,
{
    assert_eq!(offsets.len(), toptr.len() + 1);
    for (g, slot) in toptr.iter_mut().enumerate() {
        let start = offsets[g] as usize;
        let stop = offsets[g + 1] as usize;
        let mut best_idx: i64 = -1;
        // We don't need a sentinel for `best_val` because we gate on
        // `best_idx == -1` for the first comparison in each group.
        let mut best_val: Option<IN> = None;
        for (i, &val) in fromptr[start..stop].iter().enumerate() {
            let global = (start + i) as i64;
            match best_val {
                None => {
                    best_val = Some(val);
                    best_idx = global;
                }
                Some(b) if val > b => {
                    best_val = Some(val);
                    best_idx = global;
                }
                _ => {}
            }
        }
        *slot = best_idx;
=======
//! Corresponds to `src/cpu-kernels/awkward_reduce_argmax.cpp`.

/// For each group, find the index `i` of the maximum value.
/// Groups with no elements retain `-1`.
pub fn reduce_argmax<IN>(toptr: &mut [i64], fromptr: &[IN], parents: &[i64])
where
    IN: PartialOrd + Copy,
{
    assert_eq!(fromptr.len(), parents.len());
    for v in toptr.iter_mut() {
        *v = -1;
    }
    for (i, (&val, &p)) in fromptr.iter().zip(parents.iter()).enumerate() {
        let best = toptr[p as usize];
        if best == -1 || val > fromptr[best as usize] {
            toptr[p as usize] = i as i64;
        }
>>>>>>> origin/main
    }
}

macro_rules! impl_argmax {
    ($fn_name:ident, $t:ty) => {
<<<<<<< HEAD
        #[inline]
        pub fn $fn_name(toptr: &mut [i64], fromptr: &[$t], offsets: &[i64]) {
            reduce_argmax(toptr, fromptr, offsets)
=======
        pub fn $fn_name(toptr: &mut [i64], fromptr: &[$t], parents: &[i64]) {
            reduce_argmax(toptr, fromptr, parents)
>>>>>>> origin/main
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
<<<<<<< HEAD
        let offsets = [0i64, 2, 4];
        let mut out = [0i64; 2];
        reduce_argmax_int32_64(&mut out, &from, &offsets);
        assert_eq!(out, [0, 2]); // global flat indices of max in each group
    }

    #[test]
    fn empty_group_yields_minus_one() {
        let from = [5i64];
        let offsets = [0i64, 0, 1]; // group 0 empty, group 1 has element 0
        let mut out = [0i64; 2];
        reduce_argmax_int64_64(&mut out, &from, &offsets);
=======
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
>>>>>>> origin/main
        assert_eq!(out[0], -1);
        assert_eq!(out[1], 0);
    }
}
