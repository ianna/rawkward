//! Build the next-shifts array for a non-local reduction over an IndexedArray.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_IndexedArray_reduce_next_nonlocal_nextshifts_64.cpp`.

// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

/// Populate `nextshifts` with the cumulative null count seen before each
/// non-null entry in `index`.
///
/// For each position `i`:
/// * **Non-null** (`index[i] >= 0`): `nextshifts[k] = nullsum`, `k++`.
/// * **Null** (`index[i] < 0`): `nullsum++`.
///
/// This mirrors `awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64`
/// but operates on a signed index rather than a byte mask.
///
/// # Returns
///
/// The number of non-null entries written.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_reduce_next_nonlocal_nextshifts_64::indexed_array64_reduce_next_nonlocal_nextshifts_64;
///
/// // index: non-null, null, non-null, null, non-null
/// let index  = [5i64, -1, 3, -1, 7];
/// let mut shifts = [0i64; 5];
/// let n = indexed_array64_reduce_next_nonlocal_nextshifts_64(&mut shifts, &index);
/// assert_eq!(n, 3);
/// assert_eq!(&shifts[..n], &[0, 1, 2]);
/// ```
pub fn indexed_array_reduce_next_nonlocal_nextshifts_64<C>(
    nextshifts: &mut [i64],
    index: &[C],
) -> usize
where
    C: Copy + Into<i64>,
{
    let mut nullsum = 0i64;
    let mut k = 0usize;
    for &v in index {
        if v.into() >= 0 {
            nextshifts[k] = nullsum;
            k += 1;
        } else {
            nullsum += 1;
        }
    }
    k
}

pub fn indexed_array32_reduce_next_nonlocal_nextshifts_64(
    nextshifts: &mut [i64],
    index: &[i32],
) -> usize {
    indexed_array_reduce_next_nonlocal_nextshifts_64(nextshifts, index)
}
pub fn indexed_array_u32_reduce_next_nonlocal_nextshifts_64(
    nextshifts: &mut [i64],
    index: &[u32],
) -> usize {
    indexed_array_reduce_next_nonlocal_nextshifts_64(nextshifts, index)
}
pub fn indexed_array64_reduce_next_nonlocal_nextshifts_64(
    nextshifts: &mut [i64],
    index: &[i64],
) -> usize {
    indexed_array_reduce_next_nonlocal_nextshifts_64(nextshifts, index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alternating() {
        let index = [5i64, -1, 3, -1, 7];
        let mut shifts = [0i64; 5];
        let n = indexed_array64_reduce_next_nonlocal_nextshifts_64(&mut shifts, &index);
        assert_eq!(n, 3);
        assert_eq!(&shifts[..n], &[0, 1, 2]);
    }

    #[test]
    fn all_valid() {
        let index = [0i64, 1, 2, 3];
        let mut shifts = [0i64; 4];
        let n = indexed_array64_reduce_next_nonlocal_nextshifts_64(&mut shifts, &index);
        assert_eq!(n, 4);
        assert_eq!(&shifts[..n], &[0, 0, 0, 0]);
    }

    #[test]
    fn all_null() {
        let index = [-1i64; 4];
        let mut shifts = [0i64; 4];
        let n = indexed_array64_reduce_next_nonlocal_nextshifts_64(&mut shifts, &index);
        assert_eq!(n, 0);
    }

    #[test]
    fn nulls_at_start() {
        let index = [-1i64, -1, 0, 1];
        let mut shifts = [0i64; 4];
        let n = indexed_array64_reduce_next_nonlocal_nextshifts_64(&mut shifts, &index);
        assert_eq!(n, 2);
        assert_eq!(&shifts[..n], &[2, 2]);
    }

    #[test]
    fn i32_index() {
        let index = [0i32, -1, 2];
        let mut shifts = [0i64; 3];
        let n = indexed_array32_reduce_next_nonlocal_nextshifts_64(&mut shifts, &index);
        assert_eq!(n, 2);
        assert_eq!(&shifts[..n], &[0, 1]);
    }
}
