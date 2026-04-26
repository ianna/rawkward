//! Apply an array of indices to every list, producing carry and advanced arrays.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_getitem_next_array.cpp`.

// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::kernels::cpu::error::KernelError;

/// For each list `i` and each index `j` in `fromarray[0..lenarray]`:
/// * Resolve `fromarray[j]` (with negative wrap-around) against the list length.
/// * `tocarry   [i*lenarray + j] = fromstarts[i] + regular_at`
/// * `toadvanced[i*lenarray + j] = j`
///
/// # Errors
///
/// * `fromstops[i] < fromstarts[i]`
/// * `fromstops[i] > lencontent` (for non-empty lists)
/// * `fromarray[j]` out of range for list `i`
pub fn list_array_getitem_next_array<C>(
    tocarry: &mut [i64],
    toadvanced: &mut [i64],
    fromstarts: &[C],
    fromstops: &[C],
    fromarray: &[i64],
    lencontent: i64,
) -> Result<(), KernelError>
where
    C: Copy + Into<i64>,
{
    let lenstarts = fromstarts.len();
    let lenarray = fromarray.len();
    assert_eq!(fromstops.len(), lenstarts);

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
        for j in 0..lenarray {
            let mut regular_at = fromarray[j];
            if regular_at < 0 {
                regular_at += length;
            }
            if !(0 <= regular_at && regular_at < length) {
                return Err(KernelError::new(
                    "index out of range",
                    i as i64,
                    fromarray[j],
                ));
            }
            tocarry[i * lenarray + j] = start + regular_at;
            toadvanced[i * lenarray + j] = j as i64;
        }
    }
    Ok(())
}

pub fn list_array32_getitem_next_array_64(
    tocarry: &mut [i64],
    toadvanced: &mut [i64],
    fromstarts: &[i32],
    fromstops: &[i32],
    fromarray: &[i64],
    lencontent: i64,
) -> Result<(), KernelError> {
    list_array_getitem_next_array(
        tocarry, toadvanced, fromstarts, fromstops, fromarray, lencontent,
    )
}
pub fn list_array_u32_getitem_next_array_64(
    tocarry: &mut [i64],
    toadvanced: &mut [i64],
    fromstarts: &[u32],
    fromstops: &[u32],
    fromarray: &[i64],
    lencontent: i64,
) -> Result<(), KernelError> {
    list_array_getitem_next_array(
        tocarry, toadvanced, fromstarts, fromstops, fromarray, lencontent,
    )
}
pub fn list_array64_getitem_next_array_64(
    tocarry: &mut [i64],
    toadvanced: &mut [i64],
    fromstarts: &[i64],
    fromstops: &[i64],
    fromarray: &[i64],
    lencontent: i64,
) -> Result<(), KernelError> {
    list_array_getitem_next_array(
        tocarry, toadvanced, fromstarts, fromstops, fromarray, lencontent,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        // Two lists [0..3] and [3..6]; fromarray=[0,1]
        let starts = [0i64, 3];
        let stops = [3i64, 6];
        let array = [0i64, 1];
        let mut carry = [0i64; 4];
        let mut advanced = [0i64; 4];
        list_array64_getitem_next_array_64(&mut carry, &mut advanced, &starts, &stops, &array, 6)
            .unwrap();
        // list0: pos0→0, pos1→1; list1: pos0→3, pos1→4
        assert_eq!(carry, [0, 1, 3, 4]);
        assert_eq!(advanced, [0, 1, 0, 1]);
    }

    #[test]
    fn negative_index() {
        let starts = [0i64];
        let stops = [4i64];
        let array = [-1i64]; // last element → 3
        let mut carry = [0i64; 1];
        let mut advanced = [0i64; 1];
        list_array64_getitem_next_array_64(&mut carry, &mut advanced, &starts, &stops, &array, 4)
            .unwrap();
        assert_eq!(carry[0], 3);
    }

    #[test]
    fn out_of_range_error() {
        let starts = [0i64];
        let stops = [3i64];
        let array = [5i64];
        let mut carry = [0i64; 1];
        let mut advanced = [0i64; 1];
        assert!(
            list_array64_getitem_next_array_64(
                &mut carry,
                &mut advanced,
                &starts,
                &stops,
                &array,
                3
            )
            .is_err()
        );
    }
}
