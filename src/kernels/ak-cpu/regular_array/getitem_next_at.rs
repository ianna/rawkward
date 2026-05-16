// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Select a single element per row from a RegularArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_RegularArray_getitem_next_at.cpp`.

use crate::kernels::cpu::error::KernelError;

/// Resolve the integer index `at` (with negative wrap-around against `size`),
/// then for each row `i` write `tocarry[i] = i*size + regular_at`.
///
/// # Errors
///
/// Returns a [`KernelError`] if the resolved index is out of `0..size`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::regular_array_getitem_next_at::regular_array_getitem_next_at_64;
///
/// // 3 rows of size 4; at=1
/// let mut carry = [0i64; 3];
/// regular_array_getitem_next_at_64(&mut carry, 1, 3, 4).unwrap();
/// assert_eq!(carry, [1, 5, 9]);
/// ```
pub fn regular_array_getitem_next_at_64(
    tocarry: &mut [i64],
    at: i64,
    length: usize,
    size: i64,
) -> Result<(), KernelError> {
    assert!(tocarry.len() >= length);

    let mut regular_at = at;
    if regular_at < 0 {
        regular_at += size;
    }
    if !(0 <= regular_at && regular_at < size) {
        return Err(KernelError::new("index out of range", 0, at));
    }
    for i in 0..length {
        tocarry[i] = i as i64 * size + regular_at;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_at() {
        let mut carry = [0i64; 3];
        regular_array_getitem_next_at_64(&mut carry, 1, 3, 4).unwrap();
        assert_eq!(carry, [1, 5, 9]);
    }

    #[test]
    fn negative_at() {
        let mut carry = [0i64; 3];
        regular_array_getitem_next_at_64(&mut carry, -1, 3, 4).unwrap();
        assert_eq!(carry, [3, 7, 11]);
    }

    #[test]
    fn at_zero() {
        let mut carry = [0i64; 4];
        regular_array_getitem_next_at_64(&mut carry, 0, 4, 3).unwrap();
        assert_eq!(carry, [0, 3, 6, 9]);
    }

    #[test]
    fn out_of_range_error() {
        let mut carry = [0i64; 2];
        assert!(regular_array_getitem_next_at_64(&mut carry, 5, 2, 3).is_err());
    }

    #[test]
    fn negative_out_of_range() {
        let mut carry = [0i64; 2];
        assert!(regular_array_getitem_next_at_64(&mut carry, -4, 2, 3).is_err());
    }
}
