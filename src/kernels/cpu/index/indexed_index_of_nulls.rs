// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Collect the within-list positions of null entries.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_index_of_nulls.cpp`.

/// For each null (`fromindex[i] < 0`), record its position within its parent
/// list: `toindex[j++] = i - starts[parents[i]]`.
///
/// # Returns
///
/// Number of nulls written.
pub fn indexed_array_index_of_nulls<C>(
    toindex: &mut [i64],
    fromindex: &[C],
    parents: &[i64],
    starts: &[i64],
) -> usize
where
    C: Copy + Into<i64>,
{
    assert_eq!(fromindex.len(), parents.len());
    let mut j = 0usize;
    for (i, &v) in fromindex.iter().enumerate() {
        if v.into() < 0 {
            let parent = parents[i] as usize;
            toindex[j] = i as i64 - starts[parent];
            j += 1;
        }
    }
    j
}

pub fn indexed_array32_index_of_nulls(
    toindex: &mut [i64],
    fromindex: &[i32],
    parents: &[i64],
    starts: &[i64],
) -> usize {
    indexed_array_index_of_nulls(toindex, fromindex, parents, starts)
}
pub fn indexed_array_u32_index_of_nulls(
    toindex: &mut [i64],
    fromindex: &[u32],
    parents: &[i64],
    starts: &[i64],
) -> usize {
    indexed_array_index_of_nulls(toindex, fromindex, parents, starts)
}
pub fn indexed_array64_index_of_nulls(
    toindex: &mut [i64],
    fromindex: &[i64],
    parents: &[i64],
    starts: &[i64],
) -> usize {
    indexed_array_index_of_nulls(toindex, fromindex, parents, starts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        // Two lists: [0,1], [2,3]. fromindex = [5, -1, 6, -1]
        // parents = [0,0,1,1], starts = [0,2]
        // nulls at i=1 (offset=1-0=1) and i=3 (offset=3-2=1)
        let from = [5i64, -1, 6, -1];
        let parents = [0i64, 0, 1, 1];
        let starts = [0i64, 2];
        let mut out = [0i64; 4];
        let n = indexed_array64_index_of_nulls(&mut out, &from, &parents, &starts);
        assert_eq!(n, 2);
        assert_eq!(&out[..n], &[1, 1]);
    }

    #[test]
    fn no_nulls() {
        let from = [0i64, 1, 2];
        let parents = [0i64, 0, 0];
        let starts = [0i64];
        let mut out = [0i64; 3];
        let n = indexed_array64_index_of_nulls(&mut out, &from, &parents, &starts);
        assert_eq!(n, 0);
    }
}
