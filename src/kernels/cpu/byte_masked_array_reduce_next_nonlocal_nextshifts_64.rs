// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build the next-shifts array for a non-local reduction over a byte-masked array.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_64.cpp`.

/// Populate `nextshifts` with the cumulative null-count seen before each valid
/// entry in `mask`.
///
/// For each position `i`:
/// * **Valid** (`(mask[i] != 0) == validwhen`):
///   - `nextshifts[k] = nullsum`
///   - `k` is incremented.
/// * **Null**: `nullsum` is incremented.
///
/// # Returns
///
/// The number of valid entries written into `nextshifts`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::byte_masked_array_reduce_next_nonlocal_nextshifts_64::byte_masked_array_reduce_next_nonlocal_nextshifts_64;
///
/// // mask: valid, null, valid, null, valid
/// let mask = [1i8, 0, 1, 0, 1];
/// let mut shifts = [0i64; 5];
/// let n = byte_masked_array_reduce_next_nonlocal_nextshifts_64(&mut shifts, &mask, true);
/// assert_eq!(n, 3);
/// assert_eq!(&shifts[..n], &[0, 1, 2]);
/// ```
pub fn byte_masked_array_reduce_next_nonlocal_nextshifts_64(
    nextshifts: &mut [i64],
    mask: &[i8],
    validwhen: bool,
) -> usize {
    let mut nullsum = 0i64;
    let mut k = 0usize;
    for &m in mask {
        if (m != 0) == validwhen {
            nextshifts[k] = nullsum;
            k += 1;
        } else {
            nullsum += 1;
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_valid() {
        let mask = [1i8; 4];
        let mut shifts = [0i64; 4];
        let n = byte_masked_array_reduce_next_nonlocal_nextshifts_64(&mut shifts, &mask, true);
        assert_eq!(n, 4);
        assert_eq!(&shifts[..n], &[0, 0, 0, 0]);
    }

    #[test]
    fn alternating() {
        let mask = [1i8, 0, 1, 0, 1];
        let mut shifts = [0i64; 5];
        let n = byte_masked_array_reduce_next_nonlocal_nextshifts_64(&mut shifts, &mask, true);
        assert_eq!(n, 3);
        assert_eq!(&shifts[..n], &[0, 1, 2]);
    }

    #[test]
    fn all_null() {
        let mask = [0i8; 4];
        let mut shifts = [0i64; 4];
        let n = byte_masked_array_reduce_next_nonlocal_nextshifts_64(&mut shifts, &mask, true);
        assert_eq!(n, 0);
    }

    #[test]
    fn nulls_at_start() {
        let mask = [0i8, 0, 1, 1];
        let mut shifts = [0i64; 4];
        let n = byte_masked_array_reduce_next_nonlocal_nextshifts_64(&mut shifts, &mask, true);
        assert_eq!(n, 2);
        assert_eq!(&shifts[..n], &[2, 2]);
    }
}
