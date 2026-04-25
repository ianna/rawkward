// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build carry, parents, and output-index for the next reduction step.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_reduce_next_64.cpp`.

/// For each `i`:
/// * `index[i] >= 0`: `nextcarry[k]=index[i]`, `nextparents[k]=parents[i]`, `outindex[i]=k`, `k++`.
/// * `index[i] < 0`: `outindex[i]=-1`.
///
/// # Returns `usize` – number of valid entries.
pub fn indexed_array_reduce_next_64<C>(
    nextcarry: &mut [i64],
    nextparents: &mut [i64],
    outindex: &mut [i64],
    index: &[C],
    parents: &[i64],
) -> usize
where
    C: Copy + Into<i64>,
{
    assert_eq!(index.len(), parents.len());
    assert_eq!(index.len(), outindex.len());
    let mut k = 0usize;
    for (i, (&v, &p)) in index.iter().zip(parents.iter()).enumerate() {
        let j: i64 = v.into();
        if j >= 0 {
            nextcarry[k] = j;
            nextparents[k] = p;
            outindex[i] = k as i64;
            k += 1;
        } else {
            outindex[i] = -1;
        }
    }
    k
}

pub fn indexed_array32_reduce_next_64(
    nextcarry: &mut [i64],
    nextparents: &mut [i64],
    outindex: &mut [i64],
    index: &[i32],
    parents: &[i64],
) -> usize {
    indexed_array_reduce_next_64(nextcarry, nextparents, outindex, index, parents)
}
pub fn indexed_array_u32_reduce_next_64(
    nextcarry: &mut [i64],
    nextparents: &mut [i64],
    outindex: &mut [i64],
    index: &[u32],
    parents: &[i64],
) -> usize {
    indexed_array_reduce_next_64(nextcarry, nextparents, outindex, index, parents)
}
pub fn indexed_array64_reduce_next_64(
    nextcarry: &mut [i64],
    nextparents: &mut [i64],
    outindex: &mut [i64],
    index: &[i64],
    parents: &[i64],
) -> usize {
    indexed_array_reduce_next_64(nextcarry, nextparents, outindex, index, parents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed() {
        let index = [3i64, -1, 7];
        let parents = [0i64, 0, 1];
        let mut carry = [0i64; 3];
        let mut np = [0i64; 3];
        let mut outidx = [0i64; 3];
        let n = indexed_array64_reduce_next_64(&mut carry, &mut np, &mut outidx, &index, &parents);
        assert_eq!(n, 2);
        assert_eq!(&carry[..n], &[3, 7]);
        assert_eq!(&np[..n], &[0, 1]);
        assert_eq!(outidx, [0, -1, 1]);
    }

    #[test]
    fn all_null() {
        let index = [-1i64; 3];
        let parents = [0i64; 3];
        let mut carry = [0i64; 3];
        let mut np = [0i64; 3];
        let mut outidx = [0i64; 3];
        let n = indexed_array64_reduce_next_64(&mut carry, &mut np, &mut outidx, &index, &parents);
        assert_eq!(n, 0);
        assert_eq!(outidx, [-1, -1, -1]);
    }
}
