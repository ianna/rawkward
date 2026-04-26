// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Copy `fromindex` values (offset by `base`) into a slice of `toindex`.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_fill.cpp`.
//!
//! For each position `i < length`:
//! * If `fromindex[i] < 0`, write `-1`.
//! * Otherwise write `fromindex[i] + base`.

/// Generic implementation.
pub fn indexed_array_fill<FROM, TO>(
    toindex: &mut [TO],
    toindexoffset: usize,
    fromindex: &[FROM],
    base: i64,
) where
    FROM: Copy + Into<i64>,
    TO: TryFrom<i64> + Copy,
    <TO as TryFrom<i64>>::Error: std::fmt::Debug,
{
    let minus_one = TO::try_from(-1i64).expect("-1 fits");
    for (i, &v) in fromindex.iter().enumerate() {
        let fv: i64 = v.into();
        toindex[toindexoffset + i] = if fv < 0 {
            minus_one
        } else {
            TO::try_from(fv + base).expect("value fits")
        };
    }
}

/// `i32 → i64` wrapper (mirrors `awkward_IndexedArray_fill_to64_from32`).
pub fn indexed_array_fill_to64_from32(
    toindex: &mut [i64],
    toindexoffset: usize,
    fromindex: &[i32],
    base: i64,
) {
    indexed_array_fill(toindex, toindexoffset, fromindex, base);
}

/// `u32 → i64` wrapper (mirrors `awkward_IndexedArray_fill_to64_fromU32`).
pub fn indexed_array_fill_to64_fromu32(
    toindex: &mut [i64],
    toindexoffset: usize,
    fromindex: &[u32],
    base: i64,
) {
    indexed_array_fill(toindex, toindexoffset, fromindex, base);
}

/// `i64 → i64` wrapper (mirrors `awkward_IndexedArray_fill_to64_from64`).
pub fn indexed_array_fill_to64_from64(
    toindex: &mut [i64],
    toindexoffset: usize,
    fromindex: &[i64],
    base: i64,
) {
    indexed_array_fill(toindex, toindexoffset, fromindex, base);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_with_base() {
        let from = [0i32, 1, 2];
        let mut to = [0i64; 5];
        indexed_array_fill_to64_from32(&mut to, 1, &from, 10);
        assert_eq!(to, [0, 10, 11, 12, 0]);
    }

    #[test]
    fn negatives_become_minus_one() {
        let from = [-1i64, 2, -1];
        let mut to = [0i64; 3];
        indexed_array_fill_to64_from64(&mut to, 0, &from, 5);
        assert_eq!(to, [-1, 7, -1]);
    }

    #[test]
    fn u32_input() {
        let from = [0u32, 3];
        let mut to = [0i64; 2];
        indexed_array_fill_to64_fromu32(&mut to, 0, &from, 100);
        assert_eq!(to, [100, 103]);
    }
}
