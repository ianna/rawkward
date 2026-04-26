// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build the next-shifts array (with an incoming shifts array) for a
//! non-local reduction over an IndexedArray.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_IndexedArray_reduce_next_nonlocal_nextshifts_fromshifts_64.cpp`.

/// Like [`indexed_array_reduce_next_nonlocal_nextshifts_64`] but adds the
/// incoming `shifts[i]` for each non-null entry:
/// `nextshifts[k] = shifts[i] + nullsum`.
///
/// For each position `i`:
/// * **Non-null** (`index[i] >= 0`): `nextshifts[k] = shifts[i] + nullsum`, `k++`.
/// * **Null**: `nullsum++`.
///
/// # Returns
///
/// The number of non-null entries written.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_reduce_next_nonlocal_nextshifts_fromshifts_64::indexed_array64_reduce_next_nonlocal_nextshifts_fromshifts_64;
///
/// let index  = [0i64, -1, 1];
/// let shifts = [10i64, 0, 5];
/// let mut out = [0i64; 3];
/// let n = indexed_array64_reduce_next_nonlocal_nextshifts_fromshifts_64(
///     &mut out, &index, &shifts,
/// );
/// assert_eq!(n, 2);
/// assert_eq!(&out[..n], &[10, 6]); // shifts[0]+0=10, shifts[2]+1=6
/// ```
///
/// [`indexed_array_reduce_next_nonlocal_nextshifts_64`]: crate::indexed_array_reduce_next_nonlocal_nextshifts_64::indexed_array_reduce_next_nonlocal_nextshifts_64
pub fn indexed_array_reduce_next_nonlocal_nextshifts_fromshifts_64<C>(
    nextshifts: &mut [i64],
    index: &[C],
    shifts: &[i64],
) -> usize
where
    C: Copy + Into<i64>,
{
    assert_eq!(index.len(), shifts.len());
    let mut nullsum = 0i64;
    let mut k = 0usize;
    for (i, &v) in index.iter().enumerate() {
        if v.into() >= 0 {
            nextshifts[k] = shifts[i] + nullsum;
            k += 1;
        } else {
            nullsum += 1;
        }
    }
    k
}

pub fn indexed_array32_reduce_next_nonlocal_nextshifts_fromshifts_64(
    nextshifts: &mut [i64],
    index: &[i32],
    shifts: &[i64],
) -> usize {
    indexed_array_reduce_next_nonlocal_nextshifts_fromshifts_64(nextshifts, index, shifts)
}
pub fn indexed_array_u32_reduce_next_nonlocal_nextshifts_fromshifts_64(
    nextshifts: &mut [i64],
    index: &[u32],
    shifts: &[i64],
) -> usize {
    indexed_array_reduce_next_nonlocal_nextshifts_fromshifts_64(nextshifts, index, shifts)
}
pub fn indexed_array64_reduce_next_nonlocal_nextshifts_fromshifts_64(
    nextshifts: &mut [i64],
    index: &[i64],
    shifts: &[i64],
) -> usize {
    indexed_array_reduce_next_nonlocal_nextshifts_fromshifts_64(nextshifts, index, shifts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let index = [0i64, -1, 1];
        let shifts = [10i64, 0, 5];
        let mut out = [0i64; 3];
        let n = indexed_array64_reduce_next_nonlocal_nextshifts_fromshifts_64(
            &mut out, &index, &shifts,
        );
        assert_eq!(n, 2);
        assert_eq!(&out[..n], &[10, 6]);
    }

    #[test]
    fn all_valid_zero_shifts() {
        let index = [0i64, 1, 2];
        let shifts = [0i64; 3];
        let mut out = [0i64; 3];
        let n = indexed_array64_reduce_next_nonlocal_nextshifts_fromshifts_64(
            &mut out, &index, &shifts,
        );
        assert_eq!(n, 3);
        assert_eq!(&out[..n], &[0, 0, 0]);
    }

    #[test]
    fn all_null() {
        let index = [-1i64; 3];
        let shifts = [1i64, 2, 3];
        let mut out = [0i64; 3];
        let n = indexed_array64_reduce_next_nonlocal_nextshifts_fromshifts_64(
            &mut out, &index, &shifts,
        );
        assert_eq!(n, 0);
    }

    #[test]
    fn nulls_accumulate() {
        // index=[−1,−1,0]: nullsum grows to 2 before the first valid entry.
        let index = [-1i64, -1, 0];
        let shifts = [0i64, 0, 3];
        let mut out = [0i64; 3];
        let n = indexed_array64_reduce_next_nonlocal_nextshifts_fromshifts_64(
            &mut out, &index, &shifts,
        );
        assert_eq!(n, 1);
        assert_eq!(out[0], 5); // shifts[2]+nullsum = 3+2
    }

    #[test]
    fn i32_index() {
        let index = [0i32, -1, 1];
        let shifts = [2i64, 0, 4];
        let mut out = [0i64; 3];
        let n = indexed_array32_reduce_next_nonlocal_nextshifts_fromshifts_64(
            &mut out, &index, &shifts,
        );
        assert_eq!(n, 2);
        assert_eq!(&out[..n], &[2, 5]); // 2+0, 4+1
    }
}
