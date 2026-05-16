// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Count valid (non-null) entries in a jagged-slice getitem.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListArray_getitem_jagged_numvalid.cpp`.

use crate::kernels::cpu::error::KernelError;

/// Sum up the non-negative entries of `missing[slicestarts[i]..slicestops[i]]`
/// over all outer lists `i`.
///
/// Returns an error if `slicestops[i] < slicestarts[i]` or if `slicestops[i]`
/// exceeds `missing.len()`.
pub fn list_array_getitem_jagged_numvalid<T>(
    slicestarts: &[T],
    slicestops: &[T],
    missing: &[T],
) -> Result<i64, KernelError>
where
    T: Copy + Into<i64>,
{
    let missinglength = missing.len() as i64;
    let mut numvalid = 0i64;
    for (i, (&start, &stop)) in slicestarts.iter().zip(slicestops.iter()).enumerate() {
        let ss: i64 = start.into();
        let se: i64 = stop.into();
        if ss == se {
            continue;
        }
        if se < ss {
            return Err(KernelError::at(
                "jagged slice's stops[i] < starts[i]",
                i as i64,
            ));
        }
        if se > missinglength {
            return Err(KernelError::new(
                "jagged slice's offsets extend beyond its content",
                i as i64,
                se,
            ));
        }
        for j in ss..se {
            if missing[j as usize].into() >= 0 {
                numvalid += 1;
            }
        }
    }
    Ok(numvalid)
}

pub fn list_array_getitem_jagged_numvalid_64(
    slicestarts: &[i64],
    slicestops: &[i64],
    missing: &[i64],
) -> Result<i64, KernelError> {
    list_array_getitem_jagged_numvalid(slicestarts, slicestops, missing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_non_negatives() {
        let starts = [0i64, 3];
        let stops = [3i64, 5];
        let missing = [0i64, -1, 1, 2, -1];
        let n = list_array_getitem_jagged_numvalid_64(&starts, &stops, &missing).unwrap();
        assert_eq!(n, 3); // 0,1 (non-neg in 0..3) + 2 (non-neg at 3)
    }

    #[test]
    fn stop_before_start_error() {
        let starts = [3i64];
        let stops = [1i64];
        let missing = [0i64; 5];
        assert!(list_array_getitem_jagged_numvalid_64(&starts, &stops, &missing).is_err());
    }

    #[test]
    fn stop_beyond_missing_error() {
        let starts = [0i64];
        let stops = [99i64];
        let missing = [0i64; 5];
        assert!(list_array_getitem_jagged_numvalid_64(&starts, &stops, &missing).is_err());
    }
}
