//! Expand a single-offset jagged slice across every list in a ListArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_getitem_jagged_expand.cpp`.

// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::kernels::cpu::error::KernelError;

/// For each list `i` of length `jaggedsize` (verified against `fromstops[i] - fromstarts[i]`),
/// tile the single `singleoffsets[0..jaggedsize+1]` template:
///
/// * `multistarts[i*jaggedsize + j] = singleoffsets[j]`
/// * `multistops [i*jaggedsize + j] = singleoffsets[j+1]`
/// * `tocarry    [i*jaggedsize + j] = fromstarts[i] + j`
///
/// # Errors
///
/// * `stops[i] < starts[i]`
/// * `stops[i] - starts[i] ≠ jaggedsize`
pub fn list_array_getitem_jagged_expand<C>(
    multistarts: &mut [i64],
    multistops: &mut [i64],
    singleoffsets: &[i64],
    tocarry: &mut [i64],
    fromstarts: &[C],
    fromstops: &[C],
    jaggedsize: usize,
) -> Result<(), KernelError>
where
    C: Copy + Into<i64>,
{
    let length = fromstarts.len();
    assert_eq!(fromstops.len(), length);
    assert!(singleoffsets.len() > jaggedsize);

    for i in 0..length {
        let start: i64 = fromstarts[i].into();
        let stop: i64 = fromstops[i].into();
        if stop < start {
            return Err(KernelError::at("stops[i] < starts[i]", i as i64));
        }
        if (stop - start) as usize != jaggedsize {
            return Err(KernelError::at(
                "cannot fit jagged slice into nested list",
                i as i64,
            ));
        }
        for j in 0..jaggedsize {
            let out = i * jaggedsize + j;
            multistarts[out] = singleoffsets[j];
            multistops[out] = singleoffsets[j + 1];
            tocarry[out] = start + j as i64;
        }
    }
    Ok(())
}

pub fn list_array32_getitem_jagged_expand_64(
    multistarts: &mut [i64],
    multistops: &mut [i64],
    singleoffsets: &[i64],
    tocarry: &mut [i64],
    fromstarts: &[i32],
    fromstops: &[i32],
    jaggedsize: usize,
) -> Result<(), KernelError> {
    list_array_getitem_jagged_expand(
        multistarts,
        multistops,
        singleoffsets,
        tocarry,
        fromstarts,
        fromstops,
        jaggedsize,
    )
}
pub fn list_array_u32_getitem_jagged_expand_64(
    multistarts: &mut [i64],
    multistops: &mut [i64],
    singleoffsets: &[i64],
    tocarry: &mut [i64],
    fromstarts: &[u32],
    fromstops: &[u32],
    jaggedsize: usize,
) -> Result<(), KernelError> {
    list_array_getitem_jagged_expand(
        multistarts,
        multistops,
        singleoffsets,
        tocarry,
        fromstarts,
        fromstops,
        jaggedsize,
    )
}
pub fn list_array64_getitem_jagged_expand_64(
    multistarts: &mut [i64],
    multistops: &mut [i64],
    singleoffsets: &[i64],
    tocarry: &mut [i64],
    fromstarts: &[i64],
    fromstops: &[i64],
    jaggedsize: usize,
) -> Result<(), KernelError> {
    list_array_getitem_jagged_expand(
        multistarts,
        multistops,
        singleoffsets,
        tocarry,
        fromstarts,
        fromstops,
        jaggedsize,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_lists_jaggedsize_two() {
        // jaggedsize=2, singleoffsets=[0,3,7]
        // List 0: starts=0, stops=2 → out[0..2]
        // List 1: starts=5, stops=7 → out[2..4]
        let single = [0i64, 3, 7];
        let starts = [0i64, 5];
        let stops = [2i64, 7];
        let mut ms = [0i64; 4];
        let mut mp = [0i64; 4];
        let mut carry = [0i64; 4];
        list_array64_getitem_jagged_expand_64(
            &mut ms, &mut mp, &single, &mut carry, &starts, &stops, 2,
        )
        .unwrap();
        assert_eq!(ms, [0, 3, 0, 3]);
        assert_eq!(mp, [3, 7, 3, 7]);
        assert_eq!(carry, [0, 1, 5, 6]);
    }

    #[test]
    fn size_mismatch_error() {
        let single = [0i64, 1, 2, 3];
        let starts = [0i64];
        let stops = [2i64]; // length 2, not 3
        let mut ms = [0i64; 3];
        let mut mp = [0i64; 3];
        let mut carry = [0i64; 3];
        assert!(
            list_array64_getitem_jagged_expand_64(
                &mut ms, &mut mp, &single, &mut carry, &starts, &stops, 3
            )
            .is_err()
        );
    }
}
