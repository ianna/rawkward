// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Resolve and validate an index array against a RegularArray row size.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_RegularArray_getitem_next_array_regularize.cpp`.

use crate::kernels::cpu::error::KernelError;

/// For each position `j`, copy `fromarray[j]` to `toarray[j]`, applying
/// negative wrap-around (`+= size` if negative), then validate that the result
/// is in `0..size`.
///
/// # Errors
///
/// Returns a [`KernelError`] at the first out-of-range index.
///
/// # Examples
///
/// ```
/// use cpu_kernels::regular_array_getitem_next_array_regularize::regular_array_getitem_next_array_regularize_64;
///
/// let fromarray = [0i64, -1, 2];   // -1 → 2 (size=3)
/// let mut toarray = [0i64; 3];
/// regular_array_getitem_next_array_regularize_64(&mut toarray, &fromarray, 3).unwrap();
/// assert_eq!(toarray, [0, 2, 2]);
/// ```
pub fn regular_array_getitem_next_array_regularize_64(
    toarray: &mut [i64],
    fromarray: &[i64],
    size: i64,
) -> Result<(), KernelError> {
    assert_eq!(toarray.len(), fromarray.len());

    for j in 0..fromarray.len() {
        let mut v = fromarray[j];
        if v < 0 {
            v += size;
        }
        if !(0 <= v && v < size) {
            return Err(KernelError::new("index out of range", 0, fromarray[j]));
        }
        toarray[j] = v;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_with_negative() {
        let fromarray = [0i64, -1, 2];
        let mut toarray = [0i64; 3];
        regular_array_getitem_next_array_regularize_64(&mut toarray, &fromarray, 3).unwrap();
        assert_eq!(toarray, [0, 2, 2]);
    }

    #[test]
    fn all_positive_valid() {
        let fromarray = [0i64, 1, 2, 3];
        let mut toarray = [0i64; 4];
        regular_array_getitem_next_array_regularize_64(&mut toarray, &fromarray, 4).unwrap();
        assert_eq!(toarray, [0, 1, 2, 3]);
    }

    #[test]
    fn out_of_range_positive() {
        let fromarray = [5i64];
        let mut toarray = [0i64; 1];
        assert!(
            regular_array_getitem_next_array_regularize_64(&mut toarray, &fromarray, 3).is_err()
        );
    }

    #[test]
    fn out_of_range_negative() {
        let fromarray = [-4i64]; // -4 + 3 = -1 → still < 0
        let mut toarray = [0i64; 1];
        assert!(
            regular_array_getitem_next_array_regularize_64(&mut toarray, &fromarray, 3).is_err()
        );
    }

    #[test]
    fn boundary_values() {
        let fromarray = [0i64, -3]; // -3+3=0, both → 0
        let mut toarray = [0i64; 2];
        regular_array_getitem_next_array_regularize_64(&mut toarray, &fromarray, 3).unwrap();
        assert_eq!(toarray, [0, 0]);
    }
}
