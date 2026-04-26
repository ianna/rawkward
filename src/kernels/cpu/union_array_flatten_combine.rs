// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Flatten a UnionArray into flat tags, index, and offsets arrays.
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_flatten_combine.cpp`.

/// Flatten nested union content into `totags`, `toindex`, and `tooffsets`.
///
/// For each `i`, look up `tag=fromtags[i]`, `idx=fromindex[i]`, then expand
/// positions `offsetsraws[tag][idx]..offsetsraws[tag][idx+1]` into output.
pub fn union_array_flatten_combine<C, I>(
    totags: &mut [i8],
    toindex: &mut [i64],
    tooffsets: &mut [i64],
    fromtags: &[C],
    fromindex: &[I],
    offsetsraws: &[&[i64]],
) where
    C: Copy + Into<i64>,
    I: Copy + Into<i64>,
{
    assert_eq!(fromtags.len(), fromindex.len());
    assert!(tooffsets.len() > fromtags.len());
    tooffsets[0] = 0;
    let mut k = 0usize;
    for (i, (&t, &idx)) in fromtags.iter().zip(fromindex.iter()).enumerate() {
        let tag = t.into() as usize;
        let ii = idx.into() as usize;
        let offsets = offsetsraws[tag];
        let start = offsets[ii] as usize;
        let stop = offsets[ii + 1] as usize;
        tooffsets[i + 1] = tooffsets[i] + (stop - start) as i64;
        for j in start..stop {
            totags[k] = tag as i8;
            toindex[k] = j as i64;
            k += 1;
        }
    }
}

pub fn union_array32_flatten_combine_64(
    totags: &mut [i8],
    toindex: &mut [i64],
    tooffsets: &mut [i64],
    fromtags: &[i8],
    fromindex: &[i32],
    offsetsraws: &[&[i64]],
) {
    union_array_flatten_combine(totags, toindex, tooffsets, fromtags, fromindex, offsetsraws);
}
pub fn union_array64_flatten_combine_64(
    totags: &mut [i8],
    toindex: &mut [i64],
    tooffsets: &mut [i64],
    fromtags: &[i8],
    fromindex: &[i64],
    offsetsraws: &[&[i64]],
) {
    union_array_flatten_combine(totags, toindex, tooffsets, fromtags, fromindex, offsetsraws);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let offsets0 = [0i64, 2]; // content 0 has [x0, x1]
        let offsets1 = [0i64, 3]; // content 1 has [y0, y1, y2]
        let offsetsraws: &[&[i64]] = &[&offsets0, &offsets1];
        let fromtags = [0i8, 1];
        let fromindex = [0i64, 0];
        let mut totags = [0i8; 5];
        let mut toindex = [0i64; 5];
        let mut tooffsets = [0i64; 3];
        union_array64_flatten_combine_64(
            &mut totags,
            &mut toindex,
            &mut tooffsets,
            &fromtags,
            &fromindex,
            offsetsraws,
        );
        assert_eq!(tooffsets, [0, 2, 5]);
        assert_eq!(&totags[..5], &[0, 0, 1, 1, 1]);
        assert_eq!(&toindex[..5], &[0, 1, 0, 1, 2]);
    }
}
