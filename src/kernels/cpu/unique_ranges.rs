// Copyright (c) 2026 Ianna Osborne 
// SPDX-License-Identifier: BSD-3-Clause

//! Deduplicate consecutive equal elements within each group, in-place.
//!
//! Corresponds to `src/cpu-kernels/awkward_unique_ranges.cpp`.

/// For each group `i` in `fromoffsets[0..offsetslength-1]`:
/// * Seed with the first element of the group.
/// * Append successive elements only when they differ from the previous one.
/// * Write `tooffsets[i] = m` (start) and `tooffsets[i+1] = m` (after last).
///
/// `toptr` is mutated in-place — the unique values are packed to the front.
pub fn unique_ranges<T>(
    toptr: &mut [T],
    fromoffsets: &[i64],
    offsetslength: usize,
    tooffsets: &mut [i64],
) where
    T: PartialEq + Copy,
{
    assert!(tooffsets.len() >= offsetslength);
    let mut m = 0usize;
    for i in 0..offsetslength.saturating_sub(1) {
        tooffsets[i] = m as i64;
        let start = fromoffsets[i] as usize;
        let stop  = fromoffsets[i + 1] as usize;
        if start >= stop { continue; }
        toptr[m] = toptr[start];
        m += 1;
        for k in start..stop {
            if toptr[m - 1] != toptr[k] {
                toptr[m] = toptr[k];
                m += 1;
            }
        }
    }
    if offsetslength > 0 {
        tooffsets[offsetslength - 1] = m as i64;
    }
}

macro_rules! impl_unique_ranges {
    ($fn_name:ident, $t:ty) => {
        pub fn $fn_name(toptr: &mut [$t], fromoffsets: &[i64], offsetslength: usize, tooffsets: &mut [i64]) {
            unique_ranges(toptr, fromoffsets, offsetslength, tooffsets);
        }
    };
}
impl_unique_ranges!(unique_ranges_int8,    i8);
impl_unique_ranges!(unique_ranges_uint8,   u8);
impl_unique_ranges!(unique_ranges_int16,   i16);
impl_unique_ranges!(unique_ranges_uint16,  u16);
impl_unique_ranges!(unique_ranges_int32,   i32);
impl_unique_ranges!(unique_ranges_uint32,  u32);
impl_unique_ranges!(unique_ranges_int64,   i64);
impl_unique_ranges!(unique_ranges_uint64,  u64);
impl_unique_ranges!(unique_ranges_float32, f32);
impl_unique_ranges!(unique_ranges_float64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deduplicate_two_groups() {
        let mut data    = [1i32, 1, 2, 3, 3, 3];
        let fromoffsets = [0i64, 3, 6];
        let mut outoff  = [0i64; 3];
        unique_ranges_int32(&mut data, &fromoffsets, 3, &mut outoff);
        assert_eq!(&data[..outoff[2] as usize], &[1, 2, 3]);
        assert_eq!(outoff, [0, 2, 3]);
    }

    #[test]
    fn all_unique() {
        let mut data    = [1i64, 2, 3];
        let fromoffsets = [0i64, 3];
        let mut outoff  = [0i64; 2];
        unique_ranges_int64(&mut data, &fromoffsets, 2, &mut outoff);
        assert_eq!(outoff, [0, 3]);
    }

    #[test]
    fn all_same() {
        let mut data    = [5i32, 5, 5];
        let fromoffsets = [0i64, 3];
        let mut outoff  = [0i64; 2];
        unique_ranges_int32(&mut data, &fromoffsets, 2, &mut outoff);
        assert_eq!(outoff, [0, 1]);
        assert_eq!(data[0], 5);
    }
}
