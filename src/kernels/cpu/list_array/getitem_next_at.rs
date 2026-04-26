//! Carry kernel: single-integer index into each list.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_getitem_next_at.cpp`.

// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::kernels::cpu::error::KernelError;

/// For each list `i`, resolve the integer index `at` (supporting negative
/// wrap-around) and write the absolute content position into `tocarry[i]`.
///
/// Returns an error if `at` is out of range for list `i`.
pub fn list_array_getitem_next_at<C>(
    tocarry: &mut [i64],
    fromstarts: &[C],
    fromstops: &[C],
    at: i64,
) -> Result<(), KernelError>
where
    C: Copy + Into<i64>,
{
    for (i, (&start, &stop)) in fromstarts.iter().zip(fromstops.iter()).enumerate() {
        let s: i64 = start.into();
        let e: i64 = stop.into();
        let length = e - s;
        let mut regular_at = at;
        if regular_at < 0 {
            regular_at += length;
        }
        if !(0 <= regular_at && regular_at < length) {
            return Err(KernelError::new("index out of range", i as i64, at));
        }
        tocarry[i] = s + regular_at;
    }
    Ok(())
}

pub fn list_array32_getitem_next_at_64(
    tocarry: &mut [i64],
    fromstarts: &[i32],
    fromstops: &[i32],
    at: i64,
) -> Result<(), KernelError> {
    list_array_getitem_next_at(tocarry, fromstarts, fromstops, at)
}
pub fn list_array_u32_getitem_next_at_64(
    tocarry: &mut [i64],
    fromstarts: &[u32],
    fromstops: &[u32],
    at: i64,
) -> Result<(), KernelError> {
    list_array_getitem_next_at(tocarry, fromstarts, fromstops, at)
}
pub fn list_array64_getitem_next_at_64(
    tocarry: &mut [i64],
    fromstarts: &[i64],
    fromstops: &[i64],
    at: i64,
) -> Result<(), KernelError> {
    list_array_getitem_next_at(tocarry, fromstarts, fromstops, at)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_index() {
        let starts = [0i64, 3];
        let stops = [3i64, 6];
        let mut carry = [0i64; 2];
        list_array64_getitem_next_at_64(&mut carry, &starts, &stops, 1).unwrap();
        assert_eq!(carry, [1, 4]);
    }

    #[test]
    fn negative_index() {
        let starts = [0i64, 3];
        let stops = [3i64, 6];
        let mut carry = [0i64; 2];
        list_array64_getitem_next_at_64(&mut carry, &starts, &stops, -1).unwrap();
        assert_eq!(carry, [2, 5]);
    }

    #[test]
    fn out_of_range_error() {
        let starts = [0i64];
        let stops = [2i64];
        let mut carry = [0i64; 1];
        assert!(list_array64_getitem_next_at_64(&mut carry, &starts, &stops, 5).is_err());
    }

    #[test]
    fn negative_out_of_range_error() {
        let starts = [0i64];
        let stops = [2i64];
        let mut carry = [0i64; 1];
        assert!(list_array64_getitem_next_at_64(&mut carry, &starts, &stops, -3).is_err());
    }
}
