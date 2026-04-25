/ 
// Copyright (c) 2026 Ianna Osborne 
// SPDX-License-Identifier: BSD-3-Clause

//! Check whether all pairs of sub-ranges are pairwise equal.
//!
//! Corresponds to `src/cpu-kernels/awkward_NumpyArray_subrange_equal.cpp` and
//! `src/cpu-kernels/awkward_NumpyArray_subrange_equal_bool.cpp`.

/// Return `true` if every pair `(i, ii)` with `i < ii < length - 1` of
/// sub-ranges in `tmpptr` is equal.
///
/// Two sub-ranges are equal when they have the same length and every
/// corresponding element compares equal via `==`.
///
/// Note: the C++ implementation uses `i < length-1` and `ii < length-1` (not
/// `ii <= length-1`), which means the last sub-range is never compared as a
/// *right* operand.  This function preserves that behaviour.
pub fn numpy_array_subrange_equal<T>(
    tmpptr: &[T],
    fromstarts: &[i64],
    fromstops: &[i64],
    length: usize,
) -> bool
where
    T: PartialEq,
{
    let mut differ = true;
    for i in 0..length.saturating_sub(1) {
        let leftlen = (fromstops[i] - fromstarts[i]) as usize;
        for ii in i + 1..length.saturating_sub(1) {
            let rightlen = (fromstops[ii] - fromstarts[ii]) as usize;
            if leftlen == rightlen {
                differ = false;
                for j in 0..leftlen {
                    if tmpptr[fromstarts[i] as usize + j] != tmpptr[fromstarts[ii] as usize + j] {
                        differ = true;
                        break;
                    }
                }
            }
        }
    }
    !differ
}

/// Boolean variant (uses truthy comparison matching the C++ `!= 0` semantics).
pub fn numpy_array_subrange_equal_bool(
    tmpptr: &[bool],
    fromstarts: &[i64],
    fromstops: &[i64],
    length: usize,
) -> bool {
    let mut differ = true;
    for i in 0..length.saturating_sub(1) {
        let leftlen = (fromstops[i] - fromstarts[i]) as usize;
        for ii in i + 1..length.saturating_sub(1) {
            let rightlen = (fromstops[ii] - fromstarts[ii]) as usize;
            if leftlen == rightlen {
                differ = false;
                for j in 0..leftlen {
                    if (tmpptr[fromstarts[i] as usize + j] != false) !=
                       (tmpptr[fromstarts[ii] as usize + j] != false) {
                        differ = true;
                        break;
                    }
                }
            }
        }
    }
    !differ
}

macro_rules! impl_subrange_equal {
    ($fn_name:ident, $t:ty) => {
        pub fn $fn_name(tmpptr: &[$t], fromstarts: &[i64], fromstops: &[i64], length: usize) -> bool {
            numpy_array_subrange_equal(tmpptr, fromstarts, fromstops, length)
        }
    };
}
impl_subrange_equal!(numpy_array_subrange_equal_int8,    i8);
impl_subrange_equal!(numpy_array_subrange_equal_uint8,   u8);
impl_subrange_equal!(numpy_array_subrange_equal_int16,   i16);
impl_subrange_equal!(numpy_array_subrange_equal_uint16,  u16);
impl_subrange_equal!(numpy_array_subrange_equal_int32,   i32);
impl_subrange_equal!(numpy_array_subrange_equal_uint32,  u32);
impl_subrange_equal!(numpy_array_subrange_equal_int64,   i64);
impl_subrange_equal!(numpy_array_subrange_equal_uint64,  u64);
impl_subrange_equal!(numpy_array_subrange_equal_float32, f32);
impl_subrange_equal!(numpy_array_subrange_equal_float64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_ranges() {
        let data      = [1i32, 2, 3, 1, 2, 3];
        let fromstarts = [0i64, 3];
        let fromstops  = [3i64, 6];
        assert!(numpy_array_subrange_equal_int32(&data, &fromstarts, &fromstops, 3));
    }

    #[test]
    fn unequal_ranges() {
        let data       = [1i32, 2, 3, 4, 5, 6];
        let fromstarts = [0i64, 3];
        let fromstops  = [3i64, 6];
        assert!(!numpy_array_subrange_equal_int32(&data, &fromstarts, &fromstops, 3));
    }

    #[test]
    fn bool_equal() {
        let data       = [true, false, true, false];
        let fromstarts = [0i64, 2];
        let fromstops  = [2i64, 4];
        assert!(numpy_array_subrange_equal_bool(&data, &fromstarts, &fromstops, 3));
    }
}
