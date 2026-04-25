// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Apply a jagged integer-index slice to each list in a ListArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_getitem_jagged_apply.cpp`.

use crate::kernels::cpu::error::KernelError;

/// For each outer list `i`:
/// 1. Record `tooffsets[i] = k`.
/// 2. For each `j` in `slicestarts[i]..slicestops[i]`, look up
///    `index = sliceindex[j]` (supporting negative wrap-around within the
///    inner list length), compute the absolute content position, and append
///    it to `tocarry`.
///
/// Writes `tooffsets[sliceouterlen]` as the final sentinel.
///
/// # Errors
///
/// Returns [`KernelError`] if:
/// * `slicestops[i] < slicestarts[i]`
/// * `slicestops[i]` exceeds `sliceinnerlen`
/// * `fromstops[i] < fromstarts[i]`
/// * `fromstops[i]` exceeds `contentlen` (for non-empty lists)
/// * Any `sliceindex[j]` is out of range for the inner list
pub fn list_array_getitem_jagged_apply<C>(
    tooffsets: &mut [i64],
    tocarry: &mut [i64],
    slicestarts: &[i64],
    slicestops: &[i64],
    sliceindex: &[i64],
    sliceinnerlen: i64,
    fromstarts: &[C],
    fromstops: &[C],
    contentlen: i64,
) -> Result<(), KernelError>
where
    C: Copy + Into<i64>,
{
    let sliceouterlen = slicestarts.len();
    assert_eq!(slicestops.len(), sliceouterlen);
    assert_eq!(fromstarts.len(), sliceouterlen);
    assert_eq!(fromstops.len(), sliceouterlen);
    assert!(tooffsets.len() >= sliceouterlen + 1);

    let mut k = 0usize;
    for i in 0..sliceouterlen {
        let slicestart = slicestarts[i];
        let slicestop = slicestops[i];
        tooffsets[i] = k as i64;

        if slicestart != slicestop {
            if slicestop < slicestart {
                return Err(KernelError::at(
                    "jagged slice's stops[i] < starts[i]",
                    i as i64,
                ));
            }
            if slicestop > sliceinnerlen {
                return Err(KernelError::new(
                    "jagged slice's offsets extend beyond its content",
                    i as i64,
                    slicestop,
                ));
            }
            let start: i64 = fromstarts[i].into();
            let stop: i64 = fromstops[i].into();
            if stop < start {
                return Err(KernelError::at("stops[i] < starts[i]", i as i64));
            }
            if start != stop && stop > contentlen {
                return Err(KernelError::at("stops[i] > len(content)", i as i64));
            }
            let count = stop - start;
            for j in slicestart..slicestop {
                let mut index = sliceindex[j as usize];
                if index < -count || index >= count {
                    return Err(KernelError::new("index out of range", i as i64, index));
                }
                if index < 0 {
                    index += count;
                }
                tocarry[k] = start + index;
                k += 1;
            }
        }
    }
    tooffsets[sliceouterlen] = k as i64;
    Ok(())
}

pub fn list_array32_getitem_jagged_apply_64(
    tooffsets: &mut [i64],
    tocarry: &mut [i64],
    slicestarts: &[i64],
    slicestops: &[i64],
    sliceindex: &[i64],
    sliceinnerlen: i64,
    fromstarts: &[i32],
    fromstops: &[i32],
    contentlen: i64,
) -> Result<(), KernelError> {
    list_array_getitem_jagged_apply(
        tooffsets,
        tocarry,
        slicestarts,
        slicestops,
        sliceindex,
        sliceinnerlen,
        fromstarts,
        fromstops,
        contentlen,
    )
}
pub fn list_array_u32_getitem_jagged_apply_64(
    tooffsets: &mut [i64],
    tocarry: &mut [i64],
    slicestarts: &[i64],
    slicestops: &[i64],
    sliceindex: &[i64],
    sliceinnerlen: i64,
    fromstarts: &[u32],
    fromstops: &[u32],
    contentlen: i64,
) -> Result<(), KernelError> {
    list_array_getitem_jagged_apply(
        tooffsets,
        tocarry,
        slicestarts,
        slicestops,
        sliceindex,
        sliceinnerlen,
        fromstarts,
        fromstops,
        contentlen,
    )
}
pub fn list_array64_getitem_jagged_apply_64(
    tooffsets: &mut [i64],
    tocarry: &mut [i64],
    slicestarts: &[i64],
    slicestops: &[i64],
    sliceindex: &[i64],
    sliceinnerlen: i64,
    fromstarts: &[i64],
    fromstops: &[i64],
    contentlen: i64,
) -> Result<(), KernelError> {
    list_array_getitem_jagged_apply(
        tooffsets,
        tocarry,
        slicestarts,
        slicestops,
        sliceindex,
        sliceinnerlen,
        fromstarts,
        fromstops,
        contentlen,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        // Two outer lists; slice selects index 0 from each.
        // List 0: content[0..3], slice picks [0] → carry=0
        // List 1: content[3..6], slice picks [1] → carry=4
        let slicestarts = [0i64, 1];
        let slicestops = [1i64, 2];
        let sliceindex = [0i64, 1];
        let fromstarts = [0i64, 3];
        let fromstops = [3i64, 6];
        let mut offsets = [0i64; 3];
        let mut carry = [0i64; 2];
        list_array64_getitem_jagged_apply_64(
            &mut offsets,
            &mut carry,
            &slicestarts,
            &slicestops,
            &sliceindex,
            2,
            &fromstarts,
            &fromstops,
            6,
        )
        .unwrap();
        assert_eq!(offsets, [0, 1, 2]);
        assert_eq!(carry, [0, 4]);
    }

    #[test]
    fn negative_index() {
        let slicestarts = [0i64];
        let slicestops = [1i64];
        let sliceindex = [-1i64]; // last element
        let fromstarts = [0i64];
        let fromstops = [3i64];
        let mut offsets = [0i64; 2];
        let mut carry = [0i64; 1];
        list_array64_getitem_jagged_apply_64(
            &mut offsets,
            &mut carry,
            &slicestarts,
            &slicestops,
            &sliceindex,
            1,
            &fromstarts,
            &fromstops,
            3,
        )
        .unwrap();
        assert_eq!(carry[0], 2); // 0 + (-1 + 3)
    }

    #[test]
    fn empty_slice_list() {
        // slicestart == slicestop → skip
        let slicestarts = [0i64];
        let slicestops = [0i64];
        let sliceindex: [i64; 0] = [];
        let fromstarts = [0i64];
        let fromstops = [3i64];
        let mut offsets = [0i64; 2];
        let mut carry: [i64; 0] = [];
        list_array64_getitem_jagged_apply_64(
            &mut offsets,
            &mut carry,
            &slicestarts,
            &slicestops,
            &sliceindex,
            0,
            &fromstarts,
            &fromstops,
            3,
        )
        .unwrap();
        assert_eq!(offsets, [0, 0]);
    }

    #[test]
    fn index_out_of_range_error() {
        let slicestarts = [0i64];
        let slicestops = [1i64];
        let sliceindex = [10i64];
        let fromstarts = [0i64];
        let fromstops = [3i64];
        let mut offsets = [0i64; 2];
        let mut carry = [0i64; 1];
        assert!(
            list_array64_getitem_jagged_apply_64(
                &mut offsets,
                &mut carry,
                &slicestarts,
                &slicestops,
                &sliceindex,
                1,
                &fromstarts,
                &fromstops,
                3,
            )
            .is_err()
        );
    }
}
