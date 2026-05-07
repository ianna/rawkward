// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: boolean OR (any non-zero) per group.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_sum_bool.cpp`. Offsets-
//! based iteration with a per-group early-exit on the first non-zero
//! element — this is what closes the gap to awkward C++ on this kernel.

/// For each group, set `toptr[g] = true` if any
/// `fromptr[offsets[g]..offsets[g+1]] != 0`.
///
/// Per-group early-exit on the first non-zero element makes the average
/// inner-loop length roughly `2` for ~50/50 input data, regardless of the
/// nominal group size.
#[inline]
pub fn reduce_sum_bool<IN>(toptr: &mut [bool], fromptr: &[IN], offsets: &[i64])
where
    IN: PartialEq + Default + Copy,
{
    assert_eq!(offsets.len(), toptr.len() + 1);
    let zero = IN::default();
    for (g, slot) in toptr.iter_mut().enumerate() {
        let start = offsets[g] as usize;
        let stop = offsets[g + 1] as usize;
        let mut found = false;
        for &val in &fromptr[start..stop] {
            if val != zero {
                found = true;
                break;
            }
        }
        *slot = found;
    }
}

macro_rules! impl_sum_bool {
    ($fn_name:ident, $t:ty) => {
        #[inline]
        pub fn $fn_name(toptr: &mut [bool], fromptr: &[$t], offsets: &[i64]) {
            reduce_sum_bool(toptr, fromptr, offsets)
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
        let offsets = [0i64, 2, 4];
        let mut out = [false; 2];
        reduce_sum_bool_int32_64(&mut out, &from, &offsets);
        assert_eq!(out, [true, false]);
    }

    #[test]
    fn all_zero() {
        let from = [0i64; 4];
        let offsets = [0i64, 2, 4];
        let mut out = [false; 2];
        reduce_sum_bool_int64_64(&mut out, &from, &offsets);
        assert_eq!(out, [false, false]);
    }

    #[test]
    fn bool_input() {
        let from = [false, true, false];
        let offsets = [0i64, 3];
        let mut out = [false; 1];
        reduce_sum_bool_bool_64(&mut out, &from, &offsets);
        assert!(out[0]);
    }

    #[test]
    fn empty_group_is_false() {
        let from = [1i32];
        let offsets = [0i64, 1, 1];
        let mut out = [true; 2];
        reduce_sum_bool_int32_64(&mut out, &from, &offsets);
        assert_eq!(out, [true, false]);
    }
}
