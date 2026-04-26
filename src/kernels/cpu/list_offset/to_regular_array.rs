// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Determine the uniform size for converting a ListOffsetArray to a RegularArray.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_toRegularArray.cpp`.

use crate::kernels::cpu::error::KernelError;

/// Verify all lists have the same length and return it.
///
/// Returns `0` if `offsetslength <= 1` (no lists).
///
/// # Errors
///
/// * Offsets not monotonically increasing.
/// * Subarray lengths are not all equal.
pub fn list_offset_array_to_regular_array<C>(fromoffsets: &[C]) -> Result<i64, KernelError>
where
    C: Copy + Into<i64>,
{
    if fromoffsets.len() <= 1 {
        return Ok(0);
    }
    let mut size = -1i64;
    for i in 0..fromoffsets.len() - 1 {
        let count: i64 = fromoffsets[i + 1].into() - fromoffsets[i].into();
        if count < 0 {
            return Err(KernelError::at(
                "offsets must be monotonically increasing",
                i as i64,
            ));
        }
        if size == -1 {
            size = count;
        } else if size != count {
            return Err(KernelError::at(
                "cannot convert to RegularArray because subarray lengths are not regular",
                i as i64,
            ));
        }
    }
    Ok(if size == -1 { 0 } else { size })
}

pub fn list_offset_array32_to_regular_array(fromoffsets: &[i32]) -> Result<i64, KernelError> {
    list_offset_array_to_regular_array(fromoffsets)
}
pub fn list_offset_array_u32_to_regular_array(fromoffsets: &[u32]) -> Result<i64, KernelError> {
    list_offset_array_to_regular_array(fromoffsets)
}
pub fn list_offset_array64_to_regular_array(fromoffsets: &[i64]) -> Result<i64, KernelError> {
    list_offset_array_to_regular_array(fromoffsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform() {
        assert_eq!(
            list_offset_array64_to_regular_array(&[0, 3, 6, 9]).unwrap(),
            3
        );
    }

    #[test]
    fn empty_offsets() {
        assert_eq!(list_offset_array64_to_regular_array(&[]).unwrap(), 0);
    }

    #[test]
    fn single_list() {
        assert_eq!(list_offset_array64_to_regular_array(&[0, 5]).unwrap(), 5);
    }

    #[test]
    fn non_uniform_error() {
        assert!(list_offset_array64_to_regular_array(&[0, 3, 7]).is_err());
    }

    #[test]
    fn decreasing_error() {
        assert!(list_offset_array64_to_regular_array(&[5, 3]).is_err());
    }
}
