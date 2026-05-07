// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: minimum element per group.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_min.cpp`.

/// Compute the minimum element in each parent group.
///
/// `toptr` is first filled with `identity` (the "positive infinity" sentinel –
/// usually `T::MAX`), then for each `i`:
/// `toptr[parents[i]] = min(toptr[parents[i]], fromptr[i])`.
///
/// # Type parameters
///
/// * `OUT` – Output type; must support `PartialOrd` for comparison and `Copy`.
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
/// use cpu_kernels::reduce_min::reduce_min;
///
/// let from    = [3i32, 1, 5, 2];
/// let parents = [0i64, 0, 1, 1];
/// let mut out = [0i64; 2];
/// reduce_min(&mut out, &from, &parents, i64::MAX);
/// assert_eq!(out, [1, 2]);
/// ```
#[inline]
pub fn reduce_min<OUT, IN>(toptr: &mut [OUT], fromptr: &[IN], parents: &[i64], identity: OUT)
where
    OUT: PartialOrd + Copy,
    IN: Copy + Into<OUT>,
{
    assert_eq!(fromptr.len(), parents.len());
    toptr.fill(identity);
    for (&val, &p) in fromptr.iter().zip(parents.iter()) {
        let x: OUT = val.into();
        let slot = &mut toptr[p as usize];
        if x < *slot {
            *slot = x;
        }
    }
}

// ── Typed specialisation wrappers ────────────────────────────────────────────

macro_rules! impl_reduce_min {
    ($fn_name:ident, $t:ty) => {
        #[doc = concat!(
                                            "Minimum of `", stringify!($t), "` values per group."
                                        )]
        #[inline]
        pub fn $fn_name(toptr: &mut [$t], fromptr: &[$t], parents: &[i64], identity: $t) {
            reduce_min(toptr, fromptr, parents, identity)
        }
    };
}

impl_reduce_min!(reduce_min_int8_int8_64, i8);
impl_reduce_min!(reduce_min_uint8_uint8_64, u8);
impl_reduce_min!(reduce_min_int16_int16_64, i16);
impl_reduce_min!(reduce_min_uint16_uint16_64, u16);
impl_reduce_min!(reduce_min_int32_int32_64, i32);
impl_reduce_min!(reduce_min_uint32_uint32_64, u32);
impl_reduce_min!(reduce_min_int64_int64_64, i64);
impl_reduce_min!(reduce_min_uint64_uint64_64, u64);
impl_reduce_min!(reduce_min_float32_float32_64, f32);
impl_reduce_min!(reduce_min_float64_float64_64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_i64() {
        let from = [3i64, 1, 5, 2];
        let parents = [0i64, 0, 1, 1];
        let mut out = [0i64; 2];
        reduce_min_int64_int64_64(&mut out, &from, &parents, i64::MAX);
        assert_eq!(out, [1, 2]);
    }

    #[test]
    fn identity_when_no_elements() {
        let from: [i32; 0] = [];
        let parents: [i64; 0] = [];
        let mut out = [0i32; 2];
        reduce_min_int32_int32_64(&mut out, &from, &parents, i32::MAX);
        assert_eq!(out, [i32::MAX, i32::MAX]);
    }

    #[test]
    fn negative_numbers() {
        let from = [-5i32, -1, -3];
        let parents = [0i64, 0, 1];
        let mut out = [0i32; 2];
        reduce_min_int32_int32_64(&mut out, &from, &parents, i32::MAX);
        assert_eq!(out, [-5, -3]);
    }

    #[test]
    fn unsigned_min() {
        let from = [200u8, 10, 150];
        let parents = [0i64, 0, 1];
        let mut out = [0u8; 2];
        reduce_min_uint8_uint8_64(&mut out, &from, &parents, u8::MAX);
        assert_eq!(out, [10, 150]);
    }

    #[test]
    fn float_min() {
        let from = [1.5f64, -2.0, 3.0, 0.5];
        let parents = [0i64, 0, 1, 1];
        let mut out = [0.0f64; 2];
        reduce_min_float64_float64_64(&mut out, &from, &parents, f64::INFINITY);
        assert!((out[0] - (-2.0)).abs() < 1e-12);
        assert!((out[1] - 0.5).abs() < 1e-12);
    }
}
