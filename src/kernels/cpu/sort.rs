// Copyright (c) 2026 Ianna Osborne 
// SPDX-License-Identifier: BSD-3-Clause

//! Sort each group of a flat array in-place.
//!
//! Corresponds to `src/cpu-kernels/awkward_sort.cpp`.

use crate::kernels::cpu::argsort::ArgsortOrd;

/// Sort each contiguous group defined by `offsets`, writing the sorted values
/// into `toptr[0..parentslength]`.
///
/// Uses the same NaN-handling as `argsort`: NaN sorts first ascending, last descending.
pub fn sort<T>(
    toptr: &mut [T],
    fromptr: &[T],
    length: usize,
    offsets: &[i64],
    offsetslength: usize,
    parentslength: usize,
    ascending: bool,
    stable: bool,
) where
    T: ArgsortOrd + Copy,
{
    // Build a sort-permutation index over all elements.
    let mut index: Vec<usize> = (0..length).collect();

    for i in 0..offsetslength.saturating_sub(1) {
        let start = offsets[i] as usize;
        let stop  = offsets[i + 1] as usize;
        let seg   = &mut index[start..stop];

        if stable {
            if ascending {
                seg.sort_by(|&a, &b| {
                    if fromptr[a].argsort_less(&fromptr[b]) { std::cmp::Ordering::Less }
                    else if fromptr[b].argsort_less(&fromptr[a]) { std::cmp::Ordering::Greater }
                    else { std::cmp::Ordering::Equal }
                });
            } else {
                seg.sort_by(|&a, &b| {
                    if fromptr[a].argsort_greater(&fromptr[b]) { std::cmp::Ordering::Less }
                    else if fromptr[b].argsort_greater(&fromptr[a]) { std::cmp::Ordering::Greater }
                    else { std::cmp::Ordering::Equal }
                });
            }
        } else if ascending {
            seg.sort_unstable_by(|&a, &b| {
                if fromptr[a].argsort_less(&fromptr[b]) { std::cmp::Ordering::Less }
                else if fromptr[b].argsort_less(&fromptr[a]) { std::cmp::Ordering::Greater }
                else { std::cmp::Ordering::Equal }
            });
        } else {
            seg.sort_unstable_by(|&a, &b| {
                if fromptr[a].argsort_greater(&fromptr[b]) { std::cmp::Ordering::Less }
                else if fromptr[b].argsort_greater(&fromptr[a]) { std::cmp::Ordering::Greater }
                else { std::cmp::Ordering::Equal }
            });
        }
    }

    let copy_length = parentslength.min(length);
    for i in 0..copy_length {
        toptr[i] = fromptr[index[i]];
    }
}

macro_rules! impl_sort_typed {
    ($fn_name:ident, $t:ty) => {
        pub fn $fn_name(toptr: &mut [$t], fromptr: &[$t], length: usize,
                        offsets: &[i64], offsetslength: usize, parentslength: usize,
                        ascending: bool, stable: bool) {
            sort(toptr, fromptr, length, offsets, offsetslength, parentslength, ascending, stable)
        }
    };
}
impl_sort_typed!(sort_bool,    bool);
impl_sort_typed!(sort_int8,    i8);
impl_sort_typed!(sort_uint8,   u8);
impl_sort_typed!(sort_int16,   i16);
impl_sort_typed!(sort_uint16,  u16);
impl_sort_typed!(sort_int32,   i32);
impl_sort_typed!(sort_uint32,  u32);
impl_sort_typed!(sort_int64,   i64);
impl_sort_typed!(sort_uint64,  u64);
impl_sort_typed!(sort_float32, f32);
impl_sort_typed!(sort_float64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascending() {
        let from    = [3i32, 1, 2, 5, 4];
        let offsets = [0i64, 3, 5];
        let mut out = [0i32; 5];
        sort_int32(&mut out, &from, 5, &offsets, 3, 5, true, true);
        assert_eq!(out, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn descending() {
        let from    = [1i32, 3, 2];
        let offsets = [0i64, 3];
        let mut out = [0i32; 3];
        sort_int32(&mut out, &from, 3, &offsets, 2, 3, false, true);
        assert_eq!(out, [3, 2, 1]);
    }

    #[test]
    fn parentslength_limits_output() {
        let from    = [3i32, 1, 2];
        let offsets = [0i64, 3];
        let mut out = [0i32; 2];
        sort_int32(&mut out, &from, 3, &offsets, 2, 2, true, true);
        assert_eq!(out, [1, 2]);
    }
}
