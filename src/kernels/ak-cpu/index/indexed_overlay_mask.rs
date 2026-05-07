// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Overlay a byte mask on an index array, turning masked positions to -1.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_overlay_mask.cpp`.

/// For each `i`: if `mask[i]` is non-zero, write `-1`; otherwise copy `fromindex[i]`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_overlay_mask::indexed_array32_overlay_mask8_to64;
///
/// let mask      = [0i8, 1, 0, 1];
/// let fromindex = [10i32, 20, 30, 40];
/// let mut out   = [0i64; 4];
/// indexed_array32_overlay_mask8_to64(&mut out, &mask, &fromindex);
/// assert_eq!(out, [10, -1, 30, -1]);
/// ```
pub fn indexed_array_overlay_mask<C, TO>(toindex: &mut [TO], mask: &[i8], fromindex: &[C])
where
    C: Copy + Into<i64>,
    TO: TryFrom<i64> + Copy,
    <TO as TryFrom<i64>>::Error: std::fmt::Debug,
{
    assert_eq!(toindex.len(), mask.len());
    assert_eq!(toindex.len(), fromindex.len());
    let minus_one = TO::try_from(-1i64).expect("-1 fits");
    for i in 0..toindex.len() {
        toindex[i] = if mask[i] != 0 {
            minus_one
        } else {
            TO::try_from(fromindex[i].into()).expect("value fits")
        };
    }
}

pub fn indexed_array32_overlay_mask8_to64(toindex: &mut [i64], mask: &[i8], fromindex: &[i32]) {
    indexed_array_overlay_mask(toindex, mask, fromindex);
}
pub fn indexed_array_u32_overlay_mask8_to64(toindex: &mut [i64], mask: &[i8], fromindex: &[u32]) {
    indexed_array_overlay_mask(toindex, mask, fromindex);
}
pub fn indexed_array64_overlay_mask8_to64(toindex: &mut [i64], mask: &[i8], fromindex: &[i64]) {
    indexed_array_overlay_mask(toindex, mask, fromindex);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_overwrites() {
        let mask = [0i8, 1, 0, 1];
        let from = [10i32, 20, 30, 40];
        let mut out = [0i64; 4];
        indexed_array32_overlay_mask8_to64(&mut out, &mask, &from);
        assert_eq!(out, [10, -1, 30, -1]);
    }

    #[test]
    fn no_mask() {
        let mask = [0i8; 3];
        let from = [5i64, 6, 7];
        let mut out = [0i64; 3];
        indexed_array64_overlay_mask8_to64(&mut out, &mask, &from);
        assert_eq!(out, [5, 6, 7]);
    }

    #[test]
    fn all_masked() {
        let mask = [1i8; 3];
        let from = [5i64, 6, 7];
        let mut out = [0i64; 3];
        indexed_array64_overlay_mask8_to64(&mut out, &mask, &from);
        assert_eq!(out, [-1, -1, -1]);
    }
}
