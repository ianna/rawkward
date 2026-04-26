// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build the next-shifts array (with an incoming shifts array) for a
//! non-local reduction over a byte-masked array.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ByteMaskedArray_reduce_next_nonlocal_nextshifts_fromshifts_64.cpp`.

/// Like [`byte_masked_array_reduce_next_nonlocal_nextshifts_64`] but adds
/// `shifts[i]` to the cumulative null count for each valid entry.
///
/// For each position `i`:
/// * **Valid**: `nextshifts[k] = shifts[i] + nullsum`; `k++`.
/// * **Null**: `nullsum++`.
///
/// # Returns
///
/// The number of valid entries written.
///
/// # Examples
///
/// ```
/// use cpu_kernels::byte_masked_array_reduce_next_nonlocal_nextshifts_fromshifts_64::byte_masked_array_reduce_next_nonlocal_nextshifts_fromshifts_64;
///
/// let mask   = [1i8, 0, 1];
/// let shifts = [10i64, 0, 5];
/// let mut out = [0i64; 3];
/// let n = byte_masked_array_reduce_next_nonlocal_nextshifts_fromshifts_64(
///     &mut out, &mask, true, &shifts,
/// );
/// assert_eq!(n, 2);
/// assert_eq!(&out[..n], &[10, 6]); // shifts[0]+0=10, shifts[2]+1=6
/// ```
///
/// [`byte_masked_array_reduce_next_nonlocal_nextshifts_64`]: crate::byte_masked_array_reduce_next_nonlocal_nextshifts_64::byte_masked_array_reduce_next_nonlocal_nextshifts_64
pub fn byte_masked_array_reduce_next_nonlocal_nextshifts_fromshifts_64(
    nextshifts: &mut [i64],
    mask: &[i8],
    validwhen: bool,
    shifts: &[i64],
) -> usize {
    assert_eq!(mask.len(), shifts.len());
    let mut nullsum = 0i64;
    let mut k = 0usize;
    for (i, &m) in mask.iter().enumerate() {
        if (m != 0) == validwhen {
            nextshifts[k] = shifts[i] + nullsum;
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
    fn basic() {
        let mask = [1i8, 0, 1];
        let shifts = [10i64, 0, 5];
        let mut out = [0i64; 3];
        let n = byte_masked_array_reduce_next_nonlocal_nextshifts_fromshifts_64(
            &mut out, &mask, true, &shifts,
        );
        assert_eq!(n, 2);
        assert_eq!(&out[..n], &[10, 6]);
    }

    #[test]
    fn all_valid_zero_shifts() {
        let mask = [1i8; 3];
        let shifts = [0i64; 3];
        let mut out = [0i64; 3];
        let n = byte_masked_array_reduce_next_nonlocal_nextshifts_fromshifts_64(
            &mut out, &mask, true, &shifts,
        );
        assert_eq!(n, 3);
        assert_eq!(&out[..n], &[0, 0, 0]);
    }

    #[test]
    fn all_null() {
        let mask = [0i8; 3];
        let shifts = [1i64, 2, 3];
        let mut out = [0i64; 3];
        let n = byte_masked_array_reduce_next_nonlocal_nextshifts_fromshifts_64(
            &mut out, &mask, true, &shifts,
        );
        assert_eq!(n, 0);
    }
}
