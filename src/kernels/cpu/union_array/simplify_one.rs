// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Remap one variant of a UnionArray to a new tag and base-adjusted index.
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_simplify_one.cpp`.

/// For each `i` where `fromtags[i] == fromwhich`, write:
/// * `totags[i] = towhich`
/// * `toindex[i] = fromindex[i] + base`
pub fn union_array_simplify_one<FT, FI>(
    totags: &mut [i8],
    toindex: &mut [i64],
    fromtags: &[FT],
    fromindex: &[FI],
    towhich: i64,
    fromwhich: i64,
    base: i64,
) where
    FT: Copy + Into<i64>,
    FI: Copy + Into<i64>,
{
    for (i, (&t, &v)) in fromtags.iter().zip(fromindex.iter()).enumerate() {
        if t.into() == fromwhich {
            totags[i] = towhich as i8;
            toindex[i] = v.into() + base;
        }
    }
}

pub fn union_array8_32_simplify_one_to8_64(
    totags: &mut [i8],
    toindex: &mut [i64],
    fromtags: &[i8],
    fromindex: &[i32],
    towhich: i64,
    fromwhich: i64,
    base: i64,
) {
    union_array_simplify_one(
        totags, toindex, fromtags, fromindex, towhich, fromwhich, base,
    );
}
pub fn union_array8_64_simplify_one_to8_64(
    totags: &mut [i8],
    toindex: &mut [i64],
    fromtags: &[i8],
    fromindex: &[i64],
    towhich: i64,
    fromwhich: i64,
    base: i64,
) {
    union_array_simplify_one(
        totags, toindex, fromtags, fromindex, towhich, fromwhich, base,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaps_matching() {
        let fromtags = [0i8, 1, 0, 1];
        let fromindex = [5i64, 6, 7, 8];
        let mut totags = [99i8; 4];
        let mut toindex = [99i64; 4];
        union_array8_64_simplify_one_to8_64(
            &mut totags,
            &mut toindex,
            &fromtags,
            &fromindex,
            2,
            0,
            10,
        );
        assert_eq!(totags[0], 2);
        assert_eq!(toindex[0], 15);
        assert_eq!(totags[1], 99);
        assert_eq!(toindex[1], 99); // tag=1, skipped
        assert_eq!(totags[2], 2);
        assert_eq!(toindex[2], 17);
        assert_eq!(totags[3], 99);
        assert_eq!(toindex[3], 99);
    }

    #[test]
    fn no_match() {
        let fromtags = [1i8; 3];
        let fromindex = [0i64; 3];
        let mut totags = [99i8; 3];
        let mut toindex = [99i64; 3];
        union_array8_64_simplify_one_to8_64(
            &mut totags,
            &mut toindex,
            &fromtags,
            &fromindex,
            0,
            0,
            0,
        );
        assert_eq!(totags, [99, 99, 99]);
        assert_eq!(toindex, [99, 99, 99]);
    }
}
