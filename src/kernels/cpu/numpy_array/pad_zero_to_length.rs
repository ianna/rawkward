// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Pad sublists to a fixed length, zero-filling any gaps.
//!
//! Corresponds to `src/cpu-kernels/awkward_NumpyArray_pad_zero_to_length.cpp`.

/// For each sublist `k` in `fromoffsets[0..offsetslength-1]`:
/// * Copy `fromptr[fromoffsets[k]..fromoffsets[k+1]]` to `toptr[k*target..]`.
/// * Zero-fill the remaining `target - count` positions.
///
/// `toptr` must have length `(offsetslength - 1) * target`.
#[inline]
pub fn numpy_array_pad_zero_to_length<T: Copy + Default>(
    fromptr: &[T],
    fromoffsets: &[i64],
    target: usize,
    toptr: &mut [T],
) {
    let nlists = fromoffsets.len().saturating_sub(1);
    for k in 0..nlists {
        let start = fromoffsets[k] as usize;
        let end = fromoffsets[k + 1] as usize;
        let count = end - start;
        let dest = k * target;
        // copy content
        toptr[dest..dest + count].copy_from_slice(&fromptr[start..end]);
        // zero-pad remainder — `fill` compiles to memset for trivially-Copy types
        toptr[dest + count..dest + target].fill(T::default());
    }
}

pub fn numpy_array_pad_zero_to_length_uint8_int64(
    fromptr: &[u8],
    fromoffsets: &[i64],
    target: usize,
    toptr: &mut [u8],
) {
    numpy_array_pad_zero_to_length(fromptr, fromoffsets, target, toptr);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_padding() {
        // Two sublists of lengths 2 and 1, padded to target=3
        let from = [10u8, 20, 30];
        let offsets = [0i64, 2, 3];
        let mut to = [0u8; 6]; // 2 * target
        numpy_array_pad_zero_to_length_uint8_int64(&from, &offsets, 3, &mut to);
        assert_eq!(to, [10, 20, 0, 30, 0, 0]);
    }

    #[test]
    fn no_padding_needed() {
        let from = [1u8, 2, 3, 4, 5, 6];
        let offsets = [0i64, 3, 6];
        let mut to = [0u8; 6];
        numpy_array_pad_zero_to_length_uint8_int64(&from, &offsets, 3, &mut to);
        assert_eq!(to, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn empty_sublist() {
        let from = [5u8, 6];
        let offsets = [0i64, 0, 2];
        let mut to = [0u8; 4];
        numpy_array_pad_zero_to_length_uint8_int64(&from, &offsets, 2, &mut to);
        assert_eq!(to, [0, 0, 5, 6]);
    }
}
