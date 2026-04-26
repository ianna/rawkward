// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Apply an advanced (broadcasted) array index to every list.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListArray_getitem_next_array_advanced.cpp`.

use crate::kernels::cpu::error::KernelError;

/// For each list `i`, look up `fromarray[fromadvanced[i]]` (with negative
/// wrap-around), then write:
/// * `tocarry   [i] = fromstarts[i] + regular_at`
/// * `toadvanced[i] = i`
///
/// Unlike `list_array_getitem_next_array`, this picks **one** element per list
/// using the matching advanced index.
///
/// # Errors
///
/// * `fromstops[i] < fromstarts[i]`
/// * `fromstops[i] > lencontent` (for non-empty lists)
/// * Resolved index out of range
pub fn list_array_getitem_next_array_advanced<C>(
    tocarry: &mut [i64],
    toadvanced: &mut [i64],
    fromstarts: &[C],
    fromstops: &[C],
    fromarray: &[i64],
    fromadvanced: &[i64],
    lencontent: i64,
) -> Result<(), KernelError>
where
    C: Copy + Into<i64>,
{
    let lenstarts = fromstarts.len();
    assert_eq!(fromstops.len(), lenstarts);
    assert_eq!(fromadvanced.len(), lenstarts);

    for i in 0..lenstarts {
        let start: i64 = fromstarts[i].into();
        let stop: i64 = fromstops[i].into();
        if stop < start {
            return Err(KernelError::at("stops[i] < starts[i]", i as i64));
        }
        if start != stop && stop > lencontent {
            return Err(KernelError::at("stops[i] > len(content)", i as i64));
        }
        let length = stop - start;
        let raw = fromarray[fromadvanced[i] as usize];
        let mut regular_at = raw;
        if regular_at < 0 {
            regular_at += length;
        }
        if !(0 <= regular_at && regular_at < length) {
            return Err(KernelError::new("index out of range", i as i64, raw));
        }
        tocarry[i] = start + regular_at;
        toadvanced[i] = i as i64;
    }
    Ok(())
}

pub fn list_array32_getitem_next_array_advanced_64(
    tocarry: &mut [i64],
    toadvanced: &mut [i64],
    fromstarts: &[i32],
    fromstops: &[i32],
    fromarray: &[i64],
    fromadvanced: &[i64],
    lencontent: i64,
) -> Result<(), KernelError> {
    list_array_getitem_next_array_advanced(
        tocarry,
        toadvanced,
        fromstarts,
        fromstops,
        fromarray,
        fromadvanced,
        lencontent,
    )
}
pub fn list_array_u32_getitem_next_array_advanced_64(
    tocarry: &mut [i64],
    toadvanced: &mut [i64],
    fromstarts: &[u32],
    fromstops: &[u32],
    fromarray: &[i64],
    fromadvanced: &[i64],
    lencontent: i64,
) -> Result<(), KernelError> {
    list_array_getitem_next_array_advanced(
        tocarry,
        toadvanced,
        fromstarts,
        fromstops,
        fromarray,
        fromadvanced,
        lencontent,
    )
}
pub fn list_array64_getitem_next_array_advanced_64(
    tocarry: &mut [i64],
    toadvanced: &mut [i64],
    fromstarts: &[i64],
    fromstops: &[i64],
    fromarray: &[i64],
    fromadvanced: &[i64],
    lencontent: i64,
) -> Result<(), KernelError> {
    list_array_getitem_next_array_advanced(
        tocarry,
        toadvanced,
        fromstarts,
        fromstops,
        fromarray,
        fromadvanced,
        lencontent,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        // Two lists [0..3] and [3..6]; array=[1,0]; advanced=[0,1]
        // list0: array[advanced[0]]=array[0]=1 → carry=1, adv=0
        // list1: array[advanced[1]]=array[1]=0 → carry=3, adv=1
        let starts = [0i64, 3];
        let stops = [3i64, 6];
        let array = [1i64, 0];
        let advanced = [0i64, 1];
        let mut carry = [0i64; 2];
        let mut adv = [0i64; 2];
        list_array64_getitem_next_array_advanced_64(
            &mut carry, &mut adv, &starts, &stops, &array, &advanced, 6,
        )
        .unwrap();
        assert_eq!(carry, [1, 3]);
        assert_eq!(adv, [0, 1]);
    }

    #[test]
    fn negative_index() {
        let starts = [0i64];
        let stops = [4i64];
        let array = [-2i64]; // → 2
        let advanced = [0i64];
        let mut carry = [0i64; 1];
        let mut adv = [0i64; 1];
        list_array64_getitem_next_array_advanced_64(
            &mut carry, &mut adv, &starts, &stops, &array, &advanced, 4,
        )
        .unwrap();
        assert_eq!(carry[0], 2);
    }

    #[test]
    fn out_of_range_error() {
        let starts = [0i64];
        let stops = [3i64];
        let array = [10i64];
        let advanced = [0i64];
        let mut carry = [0i64; 1];
        let mut adv = [0i64; 1];
        assert!(
            list_array64_getitem_next_array_advanced_64(
                &mut carry, &mut adv, &starts, &stops, &array, &advanced, 3
            )
            .is_err()
        );
    }
}
