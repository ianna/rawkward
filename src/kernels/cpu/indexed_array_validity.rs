//! Validate the index array of an `IndexedArray` or `IndexedOptionArray`.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_validity.cpp`.

// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::kernels::cpu::error::KernelError;

/// Validate an `IndexedArray` (or `IndexedOptionArray`) index against a
/// content length.
///
/// Rules:
/// * For non-option arrays (`isoption = false`): every index must be `>= 0`.
/// * For option arrays (`isoption = true`): negative indices (`-1`) are
///   allowed and denote missing elements; only non-negative values are
///   range-checked.
/// * Every non-negative index must be `< lencontent`.
///
/// # Errors
///
/// Returns a [`KernelError`] at the first violating position.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_validity::indexed_array_validity_64;
///
/// // Non-option: all in range
/// assert!(indexed_array_validity_64(&[0, 1, 2], 3, false).is_ok());
///
/// // Option: -1 allowed
/// assert!(indexed_array_validity_64(&[0, -1, 2], 3, true).is_ok());
///
/// // Out of range
/// assert!(indexed_array_validity_64(&[5], 3, false).is_err());
/// ```
pub fn indexed_array_validity<C>(
    index: &[C],
    lencontent: i64,
    isoption: bool,
) -> Result<(), KernelError>
where
    C: PartialOrd + Copy + Into<i64>,
{
    for (i, &idx) in index.iter().enumerate() {
        let idx_i64: i64 = idx.into();

        if !isoption && idx_i64 < 0 {
            return Err(KernelError::at("index[i] < 0", i as i64));
        }
        if idx_i64 >= lencontent {
            return Err(KernelError::at("index[i] >= len(content)", i as i64));
        }
    }
    Ok(())
}

/// Typed wrapper for `i32`
/// (mirrors `awkward_IndexedArray32_validity`).
pub fn indexed_array_validity_32(
    index: &[i32],
    lencontent: i64,
    isoption: bool,
) -> Result<(), KernelError> {
    indexed_array_validity(index, lencontent, isoption)
}

/// Typed wrapper for `u32`
/// (mirrors `awkward_IndexedArrayU32_validity`).
pub fn indexed_array_validity_u32(
    index: &[u32],
    lencontent: i64,
    isoption: bool,
) -> Result<(), KernelError> {
    indexed_array_validity(index, lencontent, isoption)
}

/// Typed wrapper for `i64`
/// (mirrors `awkward_IndexedArray64_validity`).
pub fn indexed_array_validity_64(
    index: &[i64],
    lencontent: i64,
    isoption: bool,
) -> Result<(), KernelError> {
    indexed_array_validity(index, lencontent, isoption)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_option_valid() {
        assert!(indexed_array_validity_64(&[0, 1, 2], 3, false).is_ok());
    }

    #[test]
    fn non_option_negative_index() {
        let err = indexed_array_validity_64(&[-1], 3, false).unwrap_err();
        assert_eq!(err.id, 0);
        assert!(err.message.contains("index[i] < 0"));
    }

    #[test]
    fn option_allows_negative_one() {
        assert!(indexed_array_validity_64(&[0, -1, 2], 3, true).is_ok());
    }

    #[test]
    fn out_of_range() {
        let err = indexed_array_validity_64(&[5], 3, false).unwrap_err();
        assert!(err.message.contains("index[i] >= len(content)"));
    }

    #[test]
    fn out_of_range_in_option() {
        // Even in option arrays, non-negative indices must be < lencontent.
        let err = indexed_array_validity_64(&[0, 99], 3, true).unwrap_err();
        assert_eq!(err.id, 1);
    }

    #[test]
    fn empty_index_valid() {
        assert!(indexed_array_validity_64(&[], 0, false).is_ok());
    }

    #[test]
    fn u32_valid() {
        assert!(indexed_array_validity_u32(&[0u32, 1, 2], 3, false).is_ok());
    }

    #[test]
    fn error_at_third_element() {
        let err = indexed_array_validity_64(&[0, 1, 10], 5, false).unwrap_err();
        assert_eq!(err.id, 2);
    }
}
