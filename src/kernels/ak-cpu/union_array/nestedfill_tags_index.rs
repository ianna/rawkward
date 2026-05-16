// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Fill tags and index for nested content of a UnionArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_nestedfill_tags_index.cpp`.

/// For each list `i`, fill positions `tmpstarts[i]..tmpstarts[i]+fromcounts[i]`
/// with `(tag, k)` pairs, then advance `tmpstarts[i]`.
pub fn union_array_nestedfill_tags_index<T, I>(
    totags: &mut [T],
    toindex: &mut [I],
    tmpstarts: &mut [i64],
    tag: T,
    fromcounts: &[i64],
) where
    T: Copy,
    I: TryFrom<i64> + Copy,
    <I as TryFrom<i64>>::Error: std::fmt::Debug,
{
    let mut k = 0i64;
    for (i, &count) in fromcounts.iter().enumerate() {
        let start = tmpstarts[i];
        let stop = start + count;
        for j in start..stop {
            totags[j as usize] = tag;
            toindex[j as usize] = I::try_from(k).expect("k fits");
            k += 1;
        }
        tmpstarts[i] = stop;
    }
}

pub fn union_array8_64_nestedfill_tags_index_64(
    totags: &mut [i8],
    toindex: &mut [i64],
    tmpstarts: &mut [i64],
    tag: i8,
    fromcounts: &[i64],
) {
    union_array_nestedfill_tags_index(totags, toindex, tmpstarts, tag, fromcounts);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        // Two lists of lengths [2, 3]; tmpstarts = [0, 2]
        let fromcounts = [2i64, 3];
        let mut tmpstarts = [0i64, 2];
        let mut totags = [0i8; 5];
        let mut toindex = [0i64; 5];
        union_array8_64_nestedfill_tags_index_64(
            &mut totags,
            &mut toindex,
            &mut tmpstarts,
            1,
            &fromcounts,
        );
        assert_eq!(totags, [1, 1, 1, 1, 1]);
        assert_eq!(toindex, [0, 1, 2, 3, 4]);
        assert_eq!(tmpstarts, [2, 5]);
    }

    #[test]
    fn advances_tmpstarts() {
        let fromcounts = [1i64];
        let mut tmpstarts = [3i64];
        let mut totags = [0i8; 6];
        let mut toindex = [0i64; 6];
        union_array8_64_nestedfill_tags_index_64(
            &mut totags,
            &mut toindex,
            &mut tmpstarts,
            2,
            &fromcounts,
        );
        assert_eq!(totags[3], 2);
        assert_eq!(toindex[3], 0);
        assert_eq!(tmpstarts[0], 4);
    }
}
