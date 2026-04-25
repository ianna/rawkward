// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Validate the starts/stops arrays of a `ListArray`.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_validity.cpp`.

use crate::kernels::cpu::error::KernelError;

/// Validate a `ListArray`'s `starts` and `stops` arrays against a content
/// length.
///
/// For each list `i` where `start != stop` the following must hold:
/// 1. `start <= stop`
/// 2. `start >= 0`
/// 3. `stop <= lencontent`
///
/// Empty lists (`start == stop`) are always valid and are skipped.
///
/// # Type parameters
///
/// `C` is the index type (`i32`, `u32`, or `i64`).  The trait bounds ensure
/// it can be compared, cast to `i64` for error reporting, and supports
/// zero-comparison.
///
/// # Errors
///
/// Returns a [`KernelError`] identifying the first offending list index.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_array_validity::list_array_validity_64;
///
/// // Valid list array: two lists covering content positions 0..3 and 3..5
/// assert!(list_array_validity_64(&[0, 3], &[3, 5], 5).is_ok());
///
/// // Invalid: start > stop at index 0
/// assert!(list_array_validity_64(&[3], &[1], 5).is_err());
/// ```
pub fn list_array_validity<C>(starts: &[C], stops: &[C], lencontent: i64) -> Result<(), KernelError>
where
    C: PartialOrd + Copy + Into<i64>,
{
    assert_eq!(
        starts.len(),
        stops.len(),
        "starts and stops must have the same length"
    );

    let zero = {
        // Use `Into<i64>` arithmetic to obtain a typed zero.
        // We cannot write `C::default()` without an extra bound, so we rely on
        // the fact that every legal index type has 0 representable as i64 == 0.
        // We compare via Into<i64> below.
        0i64
    };

    for i in 0..starts.len() {
        let start: C = starts[i];
        let stop: C = stops[i];

        if start == stop {
            continue;
        }

        let start_i64: i64 = start.into();
        let stop_i64: i64 = stop.into();

        if start_i64 > stop_i64 {
            return Err(KernelError::at("start[i] > stop[i]", i as i64));
        }
        if start_i64 < zero {
            return Err(KernelError::at("start[i] < 0", i as i64));
        }
        if stop_i64 > lencontent {
            return Err(KernelError::at("stop[i] > len(content)", i as i64));
        }
    }
    Ok(())
}

/// Typed wrapper for `i32` starts/stops
/// (mirrors `awkward_ListArray32_validity`).
pub fn list_array_validity_32(
    starts: &[i32],
    stops: &[i32],
    lencontent: i64,
) -> Result<(), KernelError> {
    list_array_validity(starts, stops, lencontent)
}

/// Typed wrapper for `u32` starts/stops
/// (mirrors `awkward_ListArrayU32_validity`).
pub fn list_array_validity_u32(
    starts: &[u32],
    stops: &[u32],
    lencontent: i64,
) -> Result<(), KernelError> {
    list_array_validity(starts, stops, lencontent)
}

/// Typed wrapper for `i64` starts/stops
/// (mirrors `awkward_ListArray64_validity`).
pub fn list_array_validity_64(
    starts: &[i64],
    stops: &[i64],
    lencontent: i64,
) -> Result<(), KernelError> {
    list_array_validity(starts, stops, lencontent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_non_empty_lists() {
        assert!(list_array_validity_64(&[0, 3], &[3, 5], 5).is_ok());
    }

    #[test]
    fn empty_lists_always_valid() {
        // start == stop → always skipped
        assert!(list_array_validity_64(&[2, 2], &[2, 2], 0).is_ok());
    }

    #[test]
    fn start_greater_than_stop() {
        let err = list_array_validity_64(&[3], &[1], 5).unwrap_err();
        assert_eq!(err.id, 0);
        assert!(err.message.contains("start[i] > stop[i]"));
    }

    #[test]
    fn start_negative() {
        let err = list_array_validity_32(&[-1], &[2], 5).unwrap_err();
        assert_eq!(err.id, 0);
        assert!(err.message.contains("start[i] < 0"));
    }

    #[test]
    fn stop_beyond_content() {
        let err = list_array_validity_64(&[0], &[10], 5).unwrap_err();
        assert_eq!(err.id, 0);
        assert!(err.message.contains("stop[i] > len(content)"));
    }

    #[test]
    fn error_at_second_list() {
        // First list is fine, second is bad
        let err = list_array_validity_64(&[0, 0], &[1, 99], 5).unwrap_err();
        assert_eq!(err.id, 1);
    }

    #[test]
    fn u32_valid() {
        assert!(list_array_validity_u32(&[0u32, 3], &[3, 5], 5).is_ok());
    }
}
