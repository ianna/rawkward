// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Convert starts/stops to a compact offsets array.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_compact_offsets.cpp`.

use crate::kernels::cpu::error::KernelError;

/// Build `tooffsets[0..=length]` from `fromstarts` and `fromstops`.
///
/// `tooffsets[0] = 0`, then `tooffsets[i+1] = tooffsets[i] + (stops[i] - starts[i])`.
/// Returns an error if `stops[i] < starts[i]` for any `i`.
#[inline]
pub fn list_array_compact_offsets<C>(
    tooffsets: &mut [i64],
    fromstarts: &[C],
    fromstops: &[C],
) -> Result<(), KernelError>
where
    C: Copy + Into<i64>,
{
    let length = fromstarts.len();
    assert_eq!(fromstops.len(), length);
    assert!(tooffsets.len() > length);
    tooffsets[0] = 0;
    let mut acc: i64 = 0;
    for (i, (&s, &e)) in fromstarts.iter().zip(fromstops.iter()).enumerate() {
        let start: i64 = s.into();
        let stop: i64 = e.into();
        if stop < start {
            return Err(KernelError::at("stops[i] < starts[i]", i as i64));
        }
        acc += stop - start;
        tooffsets[i + 1] = acc;
    }
    Ok(())
}

#[inline]
pub fn list_array32_compact_offsets_64(
    tooffsets: &mut [i64],
    fromstarts: &[i32],
    fromstops: &[i32],
) -> Result<(), KernelError> {
    list_array_compact_offsets(tooffsets, fromstarts, fromstops)
}
#[inline]
pub fn list_array_u32_compact_offsets_64(
    tooffsets: &mut [i64],
    fromstarts: &[u32],
    fromstops: &[u32],
) -> Result<(), KernelError> {
    list_array_compact_offsets(tooffsets, fromstarts, fromstops)
}
#[inline]
pub fn list_array64_compact_offsets_64(
    tooffsets: &mut [i64],
    fromstarts: &[i64],
    fromstops: &[i64],
) -> Result<(), KernelError> {
    list_array_compact_offsets(tooffsets, fromstarts, fromstops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let starts = [0i64, 3, 5];
        let stops = [3i64, 5, 9];
        let mut out = [0i64; 4];
        list_array64_compact_offsets_64(&mut out, &starts, &stops).unwrap();
        assert_eq!(out, [0, 3, 5, 9]);
    }

    #[test]
    fn stop_before_start_error() {
        let starts = [5i64];
        let stops = [2i64];
        let mut out = [0i64; 2];
        assert!(list_array64_compact_offsets_64(&mut out, &starts, &stops).is_err());
    }

    #[test]
    fn empty_lists() {
        let starts = [2i64, 2];
        let stops = [2i64, 2];
        let mut out = [0i64; 3];
        list_array64_compact_offsets_64(&mut out, &starts, &stops).unwrap();
        assert_eq!(out, [0, 0, 0]);
    }
}
