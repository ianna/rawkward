// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Copy tags into a destination slice, adding a base offset to each value.
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_filltags.cpp`.

/// Write `totags[totagsoffset + i] = fromtags[i] + base` for each `i`.
pub fn union_array_filltags<FROM, TO>(
    totags: &mut [TO],
    totagsoffset: usize,
    fromtags: &[FROM],
    base: i64,
) where
    FROM: Copy + Into<i64>,
    TO: TryFrom<i64> + Copy,
    <TO as TryFrom<i64>>::Error: std::fmt::Debug,
{
    for (i, &v) in fromtags.iter().enumerate() {
        let val: i64 = v.into() + base;
        totags[totagsoffset + i] = TO::try_from(val).expect("value fits");
    }
}

/// `i8 → i8` wrapper (mirrors `awkward_UnionArray_filltags_to8_from8`).
pub fn union_array_filltags_to8_from8(
    totags: &mut [i8],
    totagsoffset: usize,
    fromtags: &[i8],
    base: i64,
) {
    union_array_filltags(totags, totagsoffset, fromtags, base);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_base() {
        let from = [0i8, 1, 2];
        let mut to = [0i8; 5];
        union_array_filltags_to8_from8(&mut to, 1, &from, 10);
        assert_eq!(to, [0, 10, 11, 12, 0]);
    }

    #[test]
    fn zero_base() {
        let from = [3i8, 4];
        let mut to = [0i8; 2];
        union_array_filltags_to8_from8(&mut to, 0, &from, 0);
        assert_eq!(to, [3, 4]);
    }
}
