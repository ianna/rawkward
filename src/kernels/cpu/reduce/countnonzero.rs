// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernel: count non-zero elements per group.
//!
//! Corresponds to `src/cpu-kernels/awkward_reduce_countnonzero.cpp`.

/// Count non-zero elements in each parent group.
///
/// For each input element `i`, if `fromptr[i] != 0` the counter for group
/// `parents[i]` is incremented.  `toptr` is zero-initialised before the loop.
///
/// # Type parameters
///
/// `IN` must implement `PartialEq` with a default zero value that can be
/// compared.  The trait bound `Zero` is approximated by requiring `Default`
/// (the zero/false value) and `PartialEq`.
///
/// # Panics
///
/// Panics if any `parents[i]` is out of range for `toptr`.
///
/// # Examples
///
/// ```
/// use kernels::cpu::reduce::countnonzero;
///
/// let from    = [1i32, 0, 3, 0, 5];
/// let parents = [0i64, 0, 0, 1, 1];
/// let mut out = [0i64; 2];
/// reduce_countnonzero(&mut out, &from, &parents);
/// assert_eq!(out, [2, 1]);
/// ```
#[inline]
pub fn reduce_countnonzero<IN>(toptr: &mut [i64], fromptr: &[IN], parents: &[i64])
where
    IN: PartialEq + Default,
{
    assert_eq!(fromptr.len(), parents.len());
    toptr.fill(0);
    let zero = IN::default();
    for (val, &p) in fromptr.iter().zip(parents.iter()) {
        if *val != zero {
            toptr[p as usize] += 1;
        }
    }
}

// ── Typed specialisation wrappers (mirror C++ instantiations) ───────────────
//
// `f32` and `f64` use the generic too: `Default` gives `0.0`, and
// `0.0 != 0.0` is `false` while `NaN != 0.0` is `true` — both match the
// original C++ `fromptr[i] != 0` semantics.

macro_rules! impl_countnonzero {
    ($fn_name:ident, $t:ty) => {
        #[doc = concat!("Count non-zero `", stringify!($t), "` values per group.")]
        #[inline]
        pub fn $fn_name(toptr: &mut [i64], fromptr: &[$t], parents: &[i64]) {
            reduce_countnonzero(toptr, fromptr, parents);
        }
    };
}

impl_countnonzero!(reduce_countnonzero_bool_64, bool);
impl_countnonzero!(reduce_countnonzero_int8_64, i8);
impl_countnonzero!(reduce_countnonzero_uint8_64, u8);
impl_countnonzero!(reduce_countnonzero_int16_64, i16);
impl_countnonzero!(reduce_countnonzero_uint16_64, u16);
impl_countnonzero!(reduce_countnonzero_int32_64, i32);
impl_countnonzero!(reduce_countnonzero_uint32_64, u32);
impl_countnonzero!(reduce_countnonzero_int64_64, i64);
impl_countnonzero!(reduce_countnonzero_uint64_64, u64);
impl_countnonzero!(reduce_countnonzero_float32_64, f32);
impl_countnonzero!(reduce_countnonzero_float64_64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_int() {
        let from = [1i32, 0, 3, 0, 5];
        let parents = [0i64, 0, 0, 1, 1];
        let mut out = [0i64; 2];
        reduce_countnonzero_int32_64(&mut out, &from, &parents);
        assert_eq!(out, [2, 1]);
    }

    #[test]
    fn all_zero() {
        let from = [0i64; 4];
        let parents = [0i64, 0, 1, 1];
        let mut out = [0i64; 2];
        reduce_countnonzero_int64_64(&mut out, &from, &parents);
        assert_eq!(out, [0, 0]);
    }

    #[test]
    fn bool_type() {
        let from = [true, false, true, true];
        let parents = [0i64, 0, 1, 1];
        let mut out = [0i64; 2];
        reduce_countnonzero_bool_64(&mut out, &from, &parents);
        assert_eq!(out, [1, 2]);
    }

    #[test]
    fn float32_type() {
        let from = [1.0f32, 0.0, -1.0, 0.0];
        let parents = [0i64, 0, 1, 1];
        let mut out = [0i64; 2];
        reduce_countnonzero_float32_64(&mut out, &from, &parents);
        assert_eq!(out, [1, 1]);
    }

    #[test]
    fn float64_nonzero_nan() {
        // NaN != 0.0, so it counts as non-zero (matches C++ `fromptr[i] != 0`).
        let from = [f64::NAN, 0.0];
        let parents = [0i64, 0];
        let mut out = [0i64; 1];
        reduce_countnonzero_float64_64(&mut out, &from, &parents);
        assert_eq!(out[0], 1);
    }

    #[test]
    fn output_zeroed_before_accumulation() {
        let from = [1i8];
        let parents = [0i64];
        let mut out = [99i64; 1];
        reduce_countnonzero_int8_64(&mut out, &from, &parents);
        assert_eq!(out[0], 1);
    }
}
