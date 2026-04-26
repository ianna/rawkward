// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Rpad-and-clip each list to a fixed target width, writing a flat index.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_rpad_and_clip_axis1.cpp`.

/// For each list `i` of length `rangeval = offsets[i+1] - offsets[i]`, write
/// `target` entries into `toindex[i*target .. (i+1)*target]`:
/// * Positions `0..shorter`: `offsets[i] + j` (carry the real content).
/// * Positions `shorter..target`: `-1` (null padding).
///
/// where `shorter = min(target, rangeval)`.
///
/// This clips lists that are longer than `target` and pads lists that are shorter.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_offset_array_rpad_and_clip_axis1::list_offset_array64_rpad_and_clip_axis1_64;
///
/// // offsets=[0,2,5]; target=3
/// // list0 (len 2): copy 2 real + 1 null → [0,1,-1]
/// // list1 (len 3): copy 3 real + 0 null → [2,3,4]
/// let offsets = [0i64, 2, 5];
/// let mut toindex = [0i64; 6];
/// list_offset_array64_rpad_and_clip_axis1_64(&mut toindex, &offsets, 2, 3);
/// assert_eq!(toindex, [0, 1, -1, 2, 3, 4]);
/// ```
pub fn list_offset_array_rpad_and_clip_axis1<C>(
    toindex: &mut [i64],
    fromoffsets: &[C],
    length: usize,
    target: usize,
) where
    C: Copy + Into<i64>,
{
    assert!(fromoffsets.len() > length);
    assert!(toindex.len() >= length * target);

    for i in 0..length {
        let start: i64 = fromoffsets[i].into();
        let stop: i64 = fromoffsets[i + 1].into();
        let rangeval = (stop - start) as usize;
        let shorter = target.min(rangeval);
        let out_base = i * target;
        for j in 0..shorter {
            toindex[out_base + j] = start + j as i64;
        }
        for j in shorter..target {
            toindex[out_base + j] = -1;
        }
    }
}

pub fn list_offset_array32_rpad_and_clip_axis1_64(
    toindex: &mut [i64],
    fromoffsets: &[i32],
    length: usize,
    target: usize,
) {
    list_offset_array_rpad_and_clip_axis1(toindex, fromoffsets, length, target);
}
pub fn list_offset_array_u32_rpad_and_clip_axis1_64(
    toindex: &mut [i64],
    fromoffsets: &[u32],
    length: usize,
    target: usize,
) {
    list_offset_array_rpad_and_clip_axis1(toindex, fromoffsets, length, target);
}
pub fn list_offset_array64_rpad_and_clip_axis1_64(
    toindex: &mut [i64],
    fromoffsets: &[i64],
    length: usize,
    target: usize,
) {
    list_offset_array_rpad_and_clip_axis1(toindex, fromoffsets, length, target);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_short_and_clip_none() {
        let offsets = [0i64, 2, 5];
        let mut toindex = [0i64; 6];
        list_offset_array64_rpad_and_clip_axis1_64(&mut toindex, &offsets, 2, 3);
        assert_eq!(toindex, [0, 1, -1, 2, 3, 4]);
    }

    #[test]
    fn clip_long_list() {
        // list of length 5, target=3 → take first 3
        let offsets = [0i64, 5];
        let mut toindex = [0i64; 3];
        list_offset_array64_rpad_and_clip_axis1_64(&mut toindex, &offsets, 1, 3);
        assert_eq!(toindex, [0, 1, 2]);
    }

    #[test]
    fn all_padding() {
        // Empty lists
        let offsets = [0i64, 0, 0];
        let mut toindex = [0i64; 4];
        list_offset_array64_rpad_and_clip_axis1_64(&mut toindex, &offsets, 2, 2);
        assert_eq!(toindex, [-1, -1, -1, -1]);
    }

    #[test]
    fn exact_fit() {
        let offsets = [0i64, 3, 6];
        let mut toindex = [0i64; 6];
        list_offset_array64_rpad_and_clip_axis1_64(&mut toindex, &offsets, 2, 3);
        assert_eq!(toindex, [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn i32_offsets() {
        let offsets = [0i32, 2];
        let mut toindex = [0i64; 3];
        list_offset_array32_rpad_and_clip_axis1_64(&mut toindex, &offsets, 1, 3);
        assert_eq!(toindex, [0, 1, -1]);
    }
}
