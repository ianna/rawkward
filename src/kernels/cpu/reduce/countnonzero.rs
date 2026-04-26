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
/// use cpu_kernels::reduce_countnonzero::reduce_countnonzero;
///
/// let from    = [1i32, 0, 3, 0, 5];
/// let parents = [0i64, 0, 0, 1, 1];
/// let mut out = [0i64; 2];
/// reduce_countnonzero(&mut out, &from, &parents);
/// assert_eq!(out, [2, 1]);
/// ```
pub fn reduce_countnonzero<IN>(toptr: &mut [i64], fromptr: &[IN], parents: &[i64])
where
    IN: PartialEq + Default,
{
    assert_eq!(fromptr.len(), parents.len());
    for v in toptr.iter_mut() {
        *v = 0;
    }
    let zero = IN::default();
    for (val, &p) in fromptr.iter().zip(parents.iter()) {
        if *val != zero {
            toptr[p as usize] += 1;
        }
    }
}

// ── Typed specialisation wrappers (mirror C++ instantiations) ───────────────

/// Count non-zero `bool` values per group.
pub fn reduce_countnonzero_bool_64(toptr: &mut [i64], fromptr: &[bool], parents: &[i64]) {
    reduce_countnonzero(toptr, fromptr, parents);
}
/// Count non-zero `i8` values per group.
pub fn reduce_countnonzero_int8_64(toptr: &mut [i64], fromptr: &[i8], parents: &[i64]) {
    reduce_countnonzero(toptr, fromptr, parents);
}
/// Count non-zero `u8` values per group.
pub fn reduce_countnonzero_uint8_64(toptr: &mut [i64], fromptr: &[u8], parents: &[i64]) {
    reduce_countnonzero(toptr, fromptr, parents);
}
/// Count non-zero `i16` values per group.
pub fn reduce_countnonzero_int16_64(toptr: &mut [i64], fromptr: &[i16], parents: &[i64]) {
    reduce_countnonzero(toptr, fromptr, parents);
}
/// Count non-zero `u16` values per group.
pub fn reduce_countnonzero_uint16_64(toptr: &mut [i64], fromptr: &[u16], parents: &[i64]) {
    reduce_countnonzero(toptr, fromptr, parents);
}
/// Count non-zero `i32` values per group.
pub fn reduce_countnonzero_int32_64(toptr: &mut [i64], fromptr: &[i32], parents: &[i64]) {
    reduce_countnonzero(toptr, fromptr, parents);
}
/// Count non-zero `u32` values per group.
pub fn reduce_countnonzero_uint32_64(toptr: &mut [i64], fromptr: &[u32], parents: &[i64]) {
    reduce_countnonzero(toptr, fromptr, parents);
}
/// Count non-zero `i64` values per group.
pub fn reduce_countnonzero_int64_64(toptr: &mut [i64], fromptr: &[i64], parents: &[i64]) {
    reduce_countnonzero(toptr, fromptr, parents);
}
/// Count non-zero `u64` values per group.
pub fn reduce_countnonzero_uint64_64(toptr: &mut [i64], fromptr: &[u64], parents: &[i64]) {
    reduce_countnonzero(toptr, fromptr, parents);
}
/// Count non-zero `f32` values per group.
pub fn reduce_countnonzero_float32_64(toptr: &mut [i64], fromptr: &[f32], parents: &[i64]) {
    // f32 does not implement `Default` == 0.0 that supports `PartialEq` against
    // NaN cleanly, but `0.0_f32 == 0.0_f32` is `true` and NaN is non-zero,
    // which matches C++ `fromptr[i] != 0` semantics.
    assert_eq!(fromptr.len(), parents.len());
    for v in toptr.iter_mut() {
        *v = 0;
    }
    for (&val, &p) in fromptr.iter().zip(parents.iter()) {
        if val != 0.0_f32 {
            toptr[p as usize] += 1;
        }
    }
}
/// Count non-zero `f64` values per group.
pub fn reduce_countnonzero_float64_64(toptr: &mut [i64], fromptr: &[f64], parents: &[i64]) {
    assert_eq!(fromptr.len(), parents.len());
    for v in toptr.iter_mut() {
        *v = 0;
    }
    for (&val, &p) in fromptr.iter().zip(parents.iter()) {
        if val != 0.0_f64 {
            toptr[p as usize] += 1;
        }
    }
}

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
