// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Compute the total carry length for a range slice over all lists.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListArray_getitem_next_range_carrylength.cpp`.

use crate::kernels::cpu::utils::regularize_rangeslice;

/// Sum the number of elements selected by `[start:stop:step]` across all lists.
pub fn list_array_getitem_next_range_carrylength<C>(
    fromstarts: &[C],
    fromstops: &[C],
    start: Option<i64>,
    stop: Option<i64>,
    step: i64,
) -> i64
where
    C: Copy + Into<i64>,
{
    let posstep = step > 0;
    let mut total = 0i64;
    for (&s, &e) in fromstarts.iter().zip(fromstops.iter()) {
        let length: i64 = e.into() - s.into();
        let (rs, re) = regularize_rangeslice(start, stop, posstep, length);
        let count = if posstep {
            if re > rs {
                (re - rs + step - 1) / step
            } else {
                0
            }
        } else {
            if rs > re {
                (rs - re + (-step) - 1) / (-step)
            } else {
                0
            }
        };
        total += count;
    }
    total
}

pub fn list_array64_getitem_next_range_carrylength(
    fromstarts: &[i64],
    fromstops: &[i64],
    start: Option<i64>,
    stop: Option<i64>,
    step: i64,
) -> i64 {
    list_array_getitem_next_range_carrylength(fromstarts, fromstops, start, stop, step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_slice() {
        let starts = [0i64, 3];
        let stops = [3i64, 5];
        assert_eq!(
            list_array64_getitem_next_range_carrylength(&starts, &stops, None, None, 1),
            5
        );
    }

    #[test]
    fn step_two() {
        let starts = [0i64];
        let stops = [5i64];
        assert_eq!(
            list_array64_getitem_next_range_carrylength(&starts, &stops, None, None, 2),
            3
        );
    }
}
