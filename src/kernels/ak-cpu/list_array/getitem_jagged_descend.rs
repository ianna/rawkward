// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build output offsets for descending into a jagged-slice dimension.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_getitem_jagged_descend.cpp`.

use crate::kernels::cpu::error::KernelError;

/// Verify that each slice inner-length matches the corresponding list length,
/// then build cumulative `tooffsets`.
///
/// For `sliceouterlen == 0`, `tooffsets[0] = 0`.  
/// Otherwise `tooffsets[0] = slicestarts[0]`, and for each `i`:
/// `tooffsets[i+1] = tooffsets[i] + (fromstops[i] - fromstarts[i])`.
///
/// # Errors
///
/// Returns an error if `slicestops[i] - slicestarts[i] ≠ fromstops[i] - fromstarts[i]`.
pub fn list_array_getitem_jagged_descend<C>(
    tooffsets: &mut [i64],
    slicestarts: &[i64],
    slicestops: &[i64],
    fromstarts: &[C],
    fromstops: &[C],
) -> Result<(), KernelError>
where
    C: Copy + Into<i64>,
{
    let sliceouterlen = slicestarts.len();
    assert_eq!(slicestops.len(), sliceouterlen);
    assert_eq!(fromstarts.len(), sliceouterlen);
    assert_eq!(fromstops.len(), sliceouterlen);
    assert!(tooffsets.len() > sliceouterlen);

    if sliceouterlen == 0 {
        tooffsets[0] = 0;
        return Ok(());
    }
    tooffsets[0] = slicestarts[0];
    for i in 0..sliceouterlen {
        let slicecount = slicestops[i] - slicestarts[i];
        let count: i64 = fromstops[i].into() - fromstarts[i].into();
        if slicecount != count {
            return Err(KernelError::at(
                "jagged slice inner length differs from array inner length",
                i as i64,
            ));
        }
        tooffsets[i + 1] = tooffsets[i] + count;
    }
    Ok(())
}

pub fn list_array32_getitem_jagged_descend_64(
    tooffsets: &mut [i64],
    slicestarts: &[i64],
    slicestops: &[i64],
    fromstarts: &[i32],
    fromstops: &[i32],
) -> Result<(), KernelError> {
    list_array_getitem_jagged_descend(tooffsets, slicestarts, slicestops, fromstarts, fromstops)
}
pub fn list_array_u32_getitem_jagged_descend_64(
    tooffsets: &mut [i64],
    slicestarts: &[i64],
    slicestops: &[i64],
    fromstarts: &[u32],
    fromstops: &[u32],
) -> Result<(), KernelError> {
    list_array_getitem_jagged_descend(tooffsets, slicestarts, slicestops, fromstarts, fromstops)
}
pub fn list_array64_getitem_jagged_descend_64(
    tooffsets: &mut [i64],
    slicestarts: &[i64],
    slicestops: &[i64],
    fromstarts: &[i64],
    fromstops: &[i64],
) -> Result<(), KernelError> {
    list_array_getitem_jagged_descend(tooffsets, slicestarts, slicestops, fromstarts, fromstops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        // slicestarts=[0,2], slicestops=[2,4] (lengths 2,2)
        // fromstarts=[0,3], fromstops=[2,5]   (lengths 2,2)
        let mut offsets = [0i64; 3];
        list_array64_getitem_jagged_descend_64(
            &mut offsets,
            &[0, 2],
            &[2, 4],
            &[0i64, 3],
            &[2i64, 5],
        )
        .unwrap();
        // tooffsets[0]=slicestarts[0]=0; [1]=0+2=2; [2]=2+2=4
        assert_eq!(offsets, [0, 2, 4]);
    }

    #[test]
    fn empty_outer() {
        let mut offsets = [0i64; 1];
        list_array64_getitem_jagged_descend_64(&mut offsets, &[], &[], &[], &[]).unwrap();
        assert_eq!(offsets[0], 0);
    }

    #[test]
    fn length_mismatch_error() {
        // slice says length 2, list says length 3
        let mut offsets = [0i64; 2];
        assert!(
            list_array64_getitem_jagged_descend_64(&mut offsets, &[0], &[2], &[0i64], &[3i64],)
                .is_err()
        );
    }
}
