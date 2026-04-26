// Copyright (c) 2026 Ianna Osborne 
// SPDX-License-Identifier: BSD-3-Clause

//! Reduction kernels for complex numbers represented as interleaved real/imag pairs.
//!
//! Corresponds to:
//! * `src/cpu-kernels/awkward_reduce_sum_complex.cpp`
//! * `src/cpu-kernels/awkward_reduce_argmin_complex.cpp`
//! * `src/cpu-kernels/awkward_reduce_argmax_complex.cpp`
//! * `src/cpu-kernels/awkward_reduce_countnonzero_complex.cpp`
//! * `src/cpu-kernels/awkward_reduce_max_complex.cpp`
//! * `src/cpu-kernels/awkward_reduce_min_complex.cpp`
//! * `src/cpu-kernels/awkward_reduce_prod_complex.cpp`
//! * `src/cpu-kernels/awkward_reduce_prod_bool_complex.cpp`
//! * `src/cpu-kernels/awkward_reduce_sum_bool_complex.cpp`
//!
//! All kernels represent a complex number at flat position `i` as two
//! consecutive values: `fromptr[i*2]` (real) and `fromptr[i*2+1]` (imaginary).
//! Output arrays follow the same interleaved layout.

use std::ops::AddAssign;

// ── Sum ───────────────────────────────────────────────────────────────────────

/// Sum complex numbers per group: `toptr[p*2] += re`, `toptr[p*2+1] += im`.
pub fn reduce_sum_complex<OUT, IN>(
    toptr: &mut [OUT],
    fromptr: &[IN],
    parents: &[i64],
    outlength: usize,
) where
    OUT: Default + AddAssign + Copy,
    IN: Copy + Into<OUT>,
{
    assert_eq!(fromptr.len(), parents.len() * 2);
    for i in 0..outlength { toptr[i * 2] = OUT::default(); toptr[i * 2 + 1] = OUT::default(); }
    for i in 0..parents.len() {
        let p = parents[i] as usize;
        toptr[p * 2]     += fromptr[i * 2].into();
        toptr[p * 2 + 1] += fromptr[i * 2 + 1].into();
    }
}
pub fn reduce_sum_complex64_complex64_64(toptr: &mut [f32], fromptr: &[f32], parents: &[i64], outlength: usize) { reduce_sum_complex(toptr, fromptr, parents, outlength) }
pub fn reduce_sum_complex128_complex128_64(toptr: &mut [f64], fromptr: &[f64], parents: &[i64], outlength: usize) { reduce_sum_complex(toptr, fromptr, parents, outlength) }

// ── ArgMin ────────────────────────────────────────────────────────────────────

/// Per group, find index of the element with the lexicographically smallest
/// `(real, imag)` pair.
pub fn reduce_argmin_complex<IN>(
    toptr: &mut [i64],
    fromptr: &[IN],
    parents: &[i64],
    outlength: usize,
) where
    IN: PartialOrd + Copy,
{
    for v in toptr[..outlength].iter_mut() { *v = -1; }
    for i in 0..parents.len() {
        let p = parents[i] as usize;
        let cur = toptr[p];
        let re_i = fromptr[i * 2];
        let im_i = fromptr[i * 2 + 1];
        if cur == -1 {
            toptr[p] = i as i64;
        } else {
            let re_min = fromptr[cur as usize * 2];
            let im_min = fromptr[cur as usize * 2 + 1];
            if re_i < re_min || (re_i == re_min && im_i < im_min) { toptr[p] = i as i64; }
        }
    }
}
pub fn reduce_argmin_complex64_64(toptr: &mut [i64], fromptr: &[f32], parents: &[i64], outlength: usize) { reduce_argmin_complex(toptr, fromptr, parents, outlength) }
pub fn reduce_argmin_complex128_64(toptr: &mut [i64], fromptr: &[f64], parents: &[i64], outlength: usize) { reduce_argmin_complex(toptr, fromptr, parents, outlength) }

// ── ArgMax ────────────────────────────────────────────────────────────────────

pub fn reduce_argmax_complex<IN>(toptr: &mut [i64], fromptr: &[IN], parents: &[i64], outlength: usize) where IN: PartialOrd + Copy {
    for v in toptr[..outlength].iter_mut() { *v = -1; }
    for i in 0..parents.len() {
        let p = parents[i] as usize;
        let cur = toptr[p];
        let re_i = fromptr[i * 2];
        let im_i = fromptr[i * 2 + 1];
        if cur == -1 {
            toptr[p] = i as i64;
        } else {
            let re_max = fromptr[cur as usize * 2];
            let im_max = fromptr[cur as usize * 2 + 1];
            if re_i > re_max || (re_i == re_max && im_i > im_max) { toptr[p] = i as i64; }
        }
    }
}
pub fn reduce_argmax_complex64_64(toptr: &mut [i64], fromptr: &[f32], parents: &[i64], outlength: usize) { reduce_argmax_complex(toptr, fromptr, parents, outlength) }
pub fn reduce_argmax_complex128_64(toptr: &mut [i64], fromptr: &[f64], parents: &[i64], outlength: usize) { reduce_argmax_complex(toptr, fromptr, parents, outlength) }

// ── CountNonZero ──────────────────────────────────────────────────────────────

pub fn reduce_countnonzero_complex<IN>(toptr: &mut [i64], fromptr: &[IN], parents: &[i64], outlength: usize) where IN: PartialEq + Default + Copy {
    for v in toptr[..outlength].iter_mut() { *v = 0; }
    let zero = IN::default();
    for i in 0..parents.len() {
        if fromptr[i * 2] != zero || fromptr[i * 2 + 1] != zero { toptr[parents[i] as usize] += 1; }
    }
}
pub fn reduce_countnonzero_complex64_64(toptr: &mut [i64], fromptr: &[f32], parents: &[i64], outlength: usize) {
    for v in toptr[..outlength].iter_mut() { *v = 0; }
    for i in 0..parents.len() { if fromptr[i*2] != 0.0 || fromptr[i*2+1] != 0.0 { toptr[parents[i] as usize] += 1; } }
}
pub fn reduce_countnonzero_complex128_64(toptr: &mut [i64], fromptr: &[f64], parents: &[i64], outlength: usize) {
    for v in toptr[..outlength].iter_mut() { *v = 0; }
    for i in 0..parents.len() { if fromptr[i*2] != 0.0 || fromptr[i*2+1] != 0.0 { toptr[parents[i] as usize] += 1; } }
}

// ── Max / Min ─────────────────────────────────────────────────────────────────

pub fn reduce_max_complex<T>(toptr: &mut [T], fromptr: &[T], parents: &[i64], outlength: usize, identity: T) where T: PartialOrd + Copy + DefaultZero {
    for i in 0..outlength { toptr[i*2] = identity; toptr[i*2+1] = T::default_zero(); }
    for i in 0..parents.len() {
        let p = parents[i] as usize;
        let x = fromptr[i*2]; let y = fromptr[i*2+1];
        if x > toptr[p*2] || (x == toptr[p*2] && y > toptr[p*2+1]) { toptr[p*2] = x; toptr[p*2+1] = y; }
    }
}
pub fn reduce_min_complex<T>(toptr: &mut [T], fromptr: &[T], parents: &[i64], outlength: usize, identity: T) where T: PartialOrd + Copy + DefaultZero {
    for i in 0..outlength { toptr[i*2] = identity; toptr[i*2+1] = T::default_zero(); }
    for i in 0..parents.len() {
        let p = parents[i] as usize;
        let x = fromptr[i*2]; let y = fromptr[i*2+1];
        if x < toptr[p*2] || (x == toptr[p*2] && y < toptr[p*2+1]) { toptr[p*2] = x; toptr[p*2+1] = y; }
    }
}

/// Helper for zero-initialising the imaginary part.
pub trait DefaultZero: Sized { fn default_zero() -> Self; }
impl DefaultZero for f32 { fn default_zero() -> Self { 0.0 } }
impl DefaultZero for f64 { fn default_zero() -> Self { 0.0 } }

pub fn reduce_max_complex64_complex64_64(toptr: &mut [f32], fromptr: &[f32], parents: &[i64], outlength: usize, identity: f32) { reduce_max_complex(toptr, fromptr, parents, outlength, identity) }
pub fn reduce_max_complex128_complex128_64(toptr: &mut [f64], fromptr: &[f64], parents: &[i64], outlength: usize, identity: f64) { reduce_max_complex(toptr, fromptr, parents, outlength, identity) }
pub fn reduce_min_complex64_complex64_64(toptr: &mut [f32], fromptr: &[f32], parents: &[i64], outlength: usize, identity: f32) { reduce_min_complex(toptr, fromptr, parents, outlength, identity) }
pub fn reduce_min_complex128_complex128_64(toptr: &mut [f64], fromptr: &[f64], parents: &[i64], outlength: usize, identity: f64) { reduce_min_complex(toptr, fromptr, parents, outlength, identity) }

// ── Product ───────────────────────────────────────────────────────────────────

/// Complex multiplication: `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`.
/// Accumulator initialised to `1+0i`.
pub fn reduce_prod_complex64_complex64_64(toptr: &mut [f32], fromptr: &[f32], parents: &[i64], outlength: usize) {
    for i in 0..outlength { toptr[i*2] = 1.0; toptr[i*2+1] = 0.0; }
    for i in 0..parents.len() {
        let p = parents[i] as usize;
        let (a,b,c,d) = (toptr[p*2], toptr[p*2+1], fromptr[i*2], fromptr[i*2+1]);
        toptr[p*2] = a*c - b*d; toptr[p*2+1] = a*d + b*c;
    }
}
pub fn reduce_prod_complex128_complex128_64(toptr: &mut [f64], fromptr: &[f64], parents: &[i64], outlength: usize) {
    for i in 0..outlength { toptr[i*2] = 1.0; toptr[i*2+1] = 0.0; }
    for i in 0..parents.len() {
        let p = parents[i] as usize;
        let (a,b,c,d) = (toptr[p*2], toptr[p*2+1], fromptr[i*2], fromptr[i*2+1]);
        toptr[p*2] = a*c - b*d; toptr[p*2+1] = a*d + b*c;
    }
}

// ── ProdBool / SumBool ────────────────────────────────────────────────────────

pub fn reduce_prod_bool_complex<IN>(toptr: &mut [bool], fromptr: &[IN], parents: &[i64], outlength: usize) where IN: PartialEq + Default + Copy {
    for v in toptr[..outlength].iter_mut() { *v = true; }
    let zero = IN::default();
    for i in 0..parents.len() {
        toptr[parents[i] as usize] &= fromptr[i*2] != zero || fromptr[i*2+1] != zero;
    }
}
pub fn reduce_prod_bool_complex64_64(toptr: &mut [bool], fromptr: &[f32], parents: &[i64], outlength: usize) {
    for v in toptr[..outlength].iter_mut() { *v = true; }
    for i in 0..parents.len() { toptr[parents[i] as usize] &= fromptr[i*2] != 0.0 || fromptr[i*2+1] != 0.0; }
}
pub fn reduce_prod_bool_complex128_64(toptr: &mut [bool], fromptr: &[f64], parents: &[i64], outlength: usize) {
    for v in toptr[..outlength].iter_mut() { *v = true; }
    for i in 0..parents.len() { toptr[parents[i] as usize] &= fromptr[i*2] != 0.0 || fromptr[i*2+1] != 0.0; }
}

pub fn reduce_sum_bool_complex<IN>(toptr: &mut [bool], fromptr: &[IN], parents: &[i64], outlength: usize) where IN: PartialEq + Default + Copy {
    for v in toptr[..outlength].iter_mut() { *v = false; }
    let zero = IN::default();
    for i in 0..parents.len() {
        if fromptr[i*2] != zero || fromptr[i*2+1] != zero { toptr[parents[i] as usize] = true; }
    }
}
pub fn reduce_sum_bool_complex64_64(toptr: &mut [bool], fromptr: &[f32], parents: &[i64], outlength: usize) {
    for v in toptr[..outlength].iter_mut() { *v = false; }
    for i in 0..parents.len() { if fromptr[i*2] != 0.0 || fromptr[i*2+1] != 0.0 { toptr[parents[i] as usize] = true; } }
}
pub fn reduce_sum_bool_complex128_64(toptr: &mut [bool], fromptr: &[f64], parents: &[i64], outlength: usize) {
    for v in toptr[..outlength].iter_mut() { *v = false; }
    for i in 0..parents.len() { if fromptr[i*2] != 0.0 || fromptr[i*2+1] != 0.0 { toptr[parents[i] as usize] = true; } }
}

#[cfg(test)]
mod tests {
    use super::*;

    // fromptr layout: [re0, im0, re1, im1, ...]
    fn parents2() -> Vec<i64> { vec![0, 0, 1, 1] }

    #[test]
    fn sum_complex64() {
        // (1+2i)+(3+4i) = 4+6i; (5+0i)+(0+1i) = 5+1i
        let from    = [1.0f32,2.0, 3.0,4.0, 5.0,0.0, 0.0,1.0];
        let mut out = [0.0f32; 4];
        reduce_sum_complex64_complex64_64(&mut out, &from, &parents2(), 2);
        assert!((out[0]-4.0).abs() < 1e-6);
        assert!((out[1]-6.0).abs() < 1e-6);
        assert!((out[2]-5.0).abs() < 1e-6);
        assert!((out[3]-1.0).abs() < 1e-6);
    }

    #[test]
    fn argmin_complex128() {
        // (3,1),(1,2) → min at index 1 (real 1 < 3)
        let from    = [3.0f64,1.0, 1.0,2.0];
        let parents = [0i64, 0];
        let mut out = [0i64; 1];
        reduce_argmin_complex128_64(&mut out, &from, &parents, 1);
        assert_eq!(out[0], 1);
    }

    #[test]
    fn argmax_complex64() {
        let from    = [1.0f32,0.0, 3.0,0.0];
        let parents = [0i64, 0];
        let mut out = [0i64; 1];
        reduce_argmax_complex64_64(&mut out, &from, &parents, 1);
        assert_eq!(out[0], 1);
    }

    #[test]
    fn countnonzero_complex64() {
        let from    = [1.0f32,0.0, 0.0,0.0, 0.0,1.0];
        let parents = [0i64, 0, 0];
        let mut out = [0i64; 1];
        reduce_countnonzero_complex64_64(&mut out, &from, &parents, 1);
        assert_eq!(out[0], 2);
    }

    #[test]
    fn prod_complex64() {
        // (1+0i)*(0+1i) = 0+1i
        let from    = [1.0f32,0.0, 0.0,1.0];
        let parents = [0i64, 0];
        let mut out = [0.0f32; 2];
        reduce_prod_complex64_complex64_64(&mut out, &from, &parents, 1);
        assert!((out[0]).abs() < 1e-6);
        assert!((out[1]-1.0).abs() < 1e-6);
    }

    #[test]
    fn prod_bool_complex64() {
        let from    = [1.0f32,0.0, 0.0,0.0];
        let parents = [0i64, 0];
        let mut out = [false; 1];
        reduce_prod_bool_complex64_64(&mut out, &from, &parents, 1);
        assert!(!out[0]); // one zero element → false
    }

    #[test]
    fn sum_bool_complex64() {
        let from    = [0.0f32,0.0, 1.0,0.0];
        let parents = [0i64, 0];
        let mut out = [false; 1];
        reduce_sum_bool_complex64_64(&mut out, &from, &parents, 1);
        assert!(out[0]);
    }
}
