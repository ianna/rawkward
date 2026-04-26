// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Flatten nested offsets into a single level of offsets.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_flatten_offsets.cpp`.

/// Map outer offsets through inner offsets:
/// `tooffsets[i] = inneroffsets[outeroffsets[i]]`.
pub fn list_offset_array_flatten_offsets<C>(
    tooffsets: &mut [i64],
    outeroffsets: &[C],
    inneroffsets: &[i64],
) where
    C: Copy + Into<i64>,
{
    assert_eq!(tooffsets.len(), outeroffsets.len());
    for (out, &v) in tooffsets.iter_mut().zip(outeroffsets.iter()) {
        *out = inneroffsets[v.into() as usize];
    }
}

pub fn list_offset_array32_flatten_offsets_64(
    tooffsets: &mut [i64],
    outeroffsets: &[i32],
    inneroffsets: &[i64],
) {
    list_offset_array_flatten_offsets(tooffsets, outeroffsets, inneroffsets);
}
pub fn list_offset_array_u32_flatten_offsets_64(
    tooffsets: &mut [i64],
    outeroffsets: &[u32],
    inneroffsets: &[i64],
) {
    list_offset_array_flatten_offsets(tooffsets, outeroffsets, inneroffsets);
}
pub fn list_offset_array64_flatten_offsets_64(
    tooffsets: &mut [i64],
    outeroffsets: &[i64],
    inneroffsets: &[i64],
) {
    list_offset_array_flatten_offsets(tooffsets, outeroffsets, inneroffsets);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let outer = [0i64, 2, 5];
        let inner = [0i64, 10, 20, 30, 40, 50, 60];
        let mut out = [0i64; 3];
        list_offset_array64_flatten_offsets_64(&mut out, &outer, &inner);
        assert_eq!(out, [0, 20, 50]);
    }

    #[test]
    fn identity() {
        let outer = [0i64, 1, 2, 3];
        let inner = [0i64, 5, 10, 15];
        let mut out = [0i64; 4];
        list_offset_array64_flatten_offsets_64(&mut out, &outer, &inner);
        assert_eq!(out, inner);
    }
}
