// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Apply a range slice to every list in a ListArray, building carry + offsets.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_getitem_next_range.cpp`.

use crate::kernels::cpu::error::KernelError;
use crate::kernels::cpu::utils::regularize_rangeslice;

/// For each list `i`, apply the range slice `[start:stop:step]`, write
/// absolute content positions into `tocarry`, and record cumulative counts in
/// `tooffsets`.
///
/// `start`/`stop` of `i64::MIN` mean "unspecified" (use Python defaults).
pub fn list_array_getitem_next_range<C>(
    tooffsets: &mut [C],
    tocarry: &mut [i64],
    fromstarts: &[C],
    fromstops: &[C],
    start: Option<i64>,
    stop: Option<i64>,
    step: i64,
) -> Result<(), KernelError>
where
    C: TryFrom<i64> + Copy + Into<i64>,
    <C as TryFrom<i64>>::Error: std::fmt::Debug,
{
    assert_eq!(fromstarts.len(), fromstops.len());
    assert!(tooffsets.len() > fromstarts.len());
    let posstep = step > 0;
    tooffsets[0] = C::try_from(0i64).expect("0 fits");
    let mut k = 0usize;
    for i in 0..fromstarts.len() {
        let length: i64 = fromstops[i].into() - fromstarts[i].into();
        let (rs, re) = regularize_rangeslice(start, stop, posstep, length);
        let fstart: i64 = fromstarts[i].into();
        if posstep {
            let mut j = rs;
            while j < re {
                tocarry[k] = fstart + j;
                k += 1;
                j += step;
            }
        } else {
            let mut j = rs;
            while j > re {
                tocarry[k] = fstart + j;
                k += 1;
                j += step;
            }
        }
        tooffsets[i + 1] = C::try_from(k as i64).expect("k fits");
    }
    Ok(())
}

pub fn list_array64_getitem_next_range_64(
    tooffsets: &mut [i64],
    tocarry: &mut [i64],
    fromstarts: &[i64],
    fromstops: &[i64],
    start: Option<i64>,
    stop: Option<i64>,
    step: i64,
) -> Result<(), KernelError> {
    list_array_getitem_next_range(tooffsets, tocarry, fromstarts, fromstops, start, stop, step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_slice() {
        // Two lists: [0,1,2] and [3,4]
        let starts = [0i64, 3];
        let stops = [3i64, 5];
        let mut offsets = [0i64; 3];
        let mut carry = [0i64; 10];
        list_array64_getitem_next_range_64(
            &mut offsets,
            &mut carry,
            &starts,
            &stops,
            None,
            None,
            1,
        )
        .unwrap();
        assert_eq!(offsets, [0, 3, 5]);
        assert_eq!(&carry[..5], &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn step_two() {
        let starts = [0i64];
        let stops = [5i64];
        let mut offsets = [0i64; 2];
        let mut carry = [0i64; 10];
        list_array64_getitem_next_range_64(
            &mut offsets,
            &mut carry,
            &starts,
            &stops,
            None,
            None,
            2,
        )
        .unwrap();
        // picks indices 0,2,4
        let n = offsets[1] as usize;
        assert_eq!(&carry[..n], &[0, 2, 4]);
    }

    #[test]
    fn negative_start() {
        let starts = [0i64];
        let stops = [4i64];
        let mut offsets = [0i64; 2];
        let mut carry = [0i64; 10];
        list_array64_getitem_next_range_64(
            &mut offsets,
            &mut carry,
            &starts,
            &stops,
            Some(-2),
            None,
            1,
        )
        .unwrap();
        let n = offsets[1] as usize;
        assert_eq!(&carry[..n], &[2, 3]);
    }
}
