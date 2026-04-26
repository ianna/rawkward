// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Right-pad each list to a target length, writing a sequential flat index.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListOffsetArray_rpad_axis1.cpp`.
//!
//! Unlike `rpad_and_clip`, this version does **not** clip lists that already
//! exceed `target` — it pads short lists and leaves long ones untouched.

/// For each list `i`, append to the running output index:
/// * `offsets[i]+j` for `j` in `0..rangeval` (copy real positions).
/// * `-1`           for `j` in `rangeval..target` (null padding).
///
/// A list with `rangeval > target` emits only real positions (no clipping,
/// no padding beyond `target`).
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_offset_array_rpad_axis1::list_offset_array64_rpad_axis1_64;
///
/// // offsets=[0,2,5]; target=3
/// // list0 (len 2): [0,1,-1]
/// // list1 (len 3): [2,3,4]
/// let offsets = [0i64, 2, 5];
/// let mut toindex = [0i64; 6];
/// list_offset_array64_rpad_axis1_64(&mut toindex, &offsets, 2, 3);
/// assert_eq!(toindex, [0, 1, -1, 2, 3, 4]);
/// ```
pub fn list_offset_array_rpad_axis1<C>(
    toindex: &mut [i64],
    fromoffsets: &[C],
    fromlength: usize,
    target: i64,
) where
    C: Copy + Into<i64>,
{
    assert!(fromoffsets.len() > fromlength);

    let mut count = 0usize;
    for i in 0..fromlength {
        let start: i64 = fromoffsets[i].into();
        let stop: i64 = fromoffsets[i + 1].into();
        let rangeval = stop - start;
        for j in 0..rangeval {
            toindex[count] = start + j;
            count += 1;
        }
        for _j in rangeval..target {
            toindex[count] = -1;
            count += 1;
        }
    }
}

pub fn list_offset_array32_rpad_axis1_64(
    toindex: &mut [i64],
    fromoffsets: &[i32],
    fromlength: usize,
    target: i64,
) {
    list_offset_array_rpad_axis1(toindex, fromoffsets, fromlength, target);
}
pub fn list_offset_array_u32_rpad_axis1_64(
    toindex: &mut [i64],
    fromoffsets: &[u32],
    fromlength: usize,
    target: i64,
) {
    list_offset_array_rpad_axis1(toindex, fromoffsets, fromlength, target);
}
pub fn list_offset_array64_rpad_axis1_64(
    toindex: &mut [i64],
    fromoffsets: &[i64],
    fromlength: usize,
    target: i64,
) {
    list_offset_array_rpad_axis1(toindex, fromoffsets, fromlength, target);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_short_list() {
        let offsets = [0i64, 2, 5];
        let mut toindex = [0i64; 6];
        list_offset_array64_rpad_axis1_64(&mut toindex, &offsets, 2, 3);
        assert_eq!(toindex, [0, 1, -1, 2, 3, 4]);
    }

    #[test]
    fn list_longer_than_target_not_clipped() {
        // len=5, target=3 → emits all 5 real positions (no clipping)
        let offsets = [0i64, 5];
        let mut toindex = [0i64; 5];
        list_offset_array64_rpad_axis1_64(&mut toindex, &offsets, 1, 3);
        assert_eq!(toindex, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn empty_list_fully_padded() {
        let offsets = [0i64, 0];
        let mut toindex = [0i64; 3];
        list_offset_array64_rpad_axis1_64(&mut toindex, &offsets, 1, 3);
        assert_eq!(toindex, [-1, -1, -1]);
    }

    #[test]
    fn exact_fit() {
        let offsets = [0i64, 3];
        let mut toindex = [0i64; 3];
        list_offset_array64_rpad_axis1_64(&mut toindex, &offsets, 1, 3);
        assert_eq!(toindex, [0, 1, 2]);
    }

    #[test]
    fn i32_offsets() {
        let offsets = [0i32, 1, 3];
        let mut toindex = [0i64; 4];
        list_offset_array32_rpad_axis1_64(&mut toindex, &offsets, 2, 2);
        assert_eq!(toindex, [0, -1, 1, 2]);
    }
}
