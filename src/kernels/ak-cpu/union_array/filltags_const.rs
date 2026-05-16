// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Fill a slice of a tags array with a constant value.
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_filltags_const.cpp`.

/// Write `totags[totagsoffset + i] = base` for `i` in `0..length`.
pub fn union_array_filltags_const(totags: &mut [i8], totagsoffset: usize, length: usize, base: i8) {
    for i in 0..length {
        totags[totagsoffset + i] = base;
    }
}

/// Alias matching `awkward_UnionArray_filltags_to8_const`.
pub fn union_array_filltags_to8_const(
    totags: &mut [i8],
    totagsoffset: usize,
    length: usize,
    base: i8,
) {
    union_array_filltags_const(totags, totagsoffset, length, base);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_slice() {
        let mut tags = [0i8; 6];
        union_array_filltags_to8_const(&mut tags, 2, 3, 7);
        assert_eq!(tags, [0, 0, 7, 7, 7, 0]);
    }

    #[test]
    fn zero_length() {
        let mut tags = [5i8; 3];
        union_array_filltags_to8_const(&mut tags, 1, 0, 9);
        assert_eq!(tags, [5, 5, 5]);
    }
}
