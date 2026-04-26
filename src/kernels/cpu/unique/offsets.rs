// Copyright (c) 2026 Ianna Osborne 
// SPDX-License-Identifier: BSD-3-Clause

//! Build output offsets for a unique-values pass, handling empty groups.
//!
//! Corresponds to `src/cpu-kernels/awkward_unique_offsets.cpp`.

/// Populate `tooffsets[0..startslength+1]` from `fromoffsets` and `starts`.
///
/// For each `i` in `0..length`, write `tooffsets[j] = fromoffsets[i]` and
/// advance `j`.  When consecutive `starts[j] == starts[j+1]`, emit an extra
/// repeat entry for the empty group before advancing.  Finally write
/// `tooffsets[startslength] = fromoffsets[length-1]`.
pub fn unique_offsets<T>(
    tooffsets: &mut [T],
    length: usize,
    fromoffsets: &[i64],
    starts: &[i64],
    startslength: usize,
) where
    T: TryFrom<i64> + Copy,
    <T as TryFrom<i64>>::Error: std::fmt::Debug,
{
    let mut j = 0usize;
    for i in 0..length {
        tooffsets[j] = T::try_from(fromoffsets[i]).expect("fits");
        while j + 1 < startslength && starts[j] == starts[j + 1] {
            tooffsets[j + 1] = T::try_from(fromoffsets[i]).expect("fits");
            j += 1;
        }
        j += 1;
    }
    if startslength < tooffsets.len() && length > 0 {
        tooffsets[startslength] = T::try_from(fromoffsets[length - 1]).expect("fits");
    }
}

pub fn unique_offsets_int8(tooffsets: &mut [i8], length: usize, fromoffsets: &[i64], starts: &[i64], startslength: usize) {
    unique_offsets(tooffsets, length, fromoffsets, starts, startslength);
}
pub fn unique_offsets_int16(tooffsets: &mut [i16], length: usize, fromoffsets: &[i64], starts: &[i64], startslength: usize) {
    unique_offsets(tooffsets, length, fromoffsets, starts, startslength);
}
pub fn unique_offsets_int32(tooffsets: &mut [i32], length: usize, fromoffsets: &[i64], starts: &[i64], startslength: usize) {
    unique_offsets(tooffsets, length, fromoffsets, starts, startslength);
}
pub fn unique_offsets_int64(tooffsets: &mut [i64], length: usize, fromoffsets: &[i64], starts: &[i64], startslength: usize) {
    unique_offsets(tooffsets, length, fromoffsets, starts, startslength);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let fromoffsets = [0i64, 3, 6, 9];
        let starts      = [0i64, 3, 6];
        let mut out     = [0i64; 4];
        unique_offsets_int64(&mut out, 3, &fromoffsets, &starts, 3);
        assert_eq!(out, [0, 3, 6, 6]);
    }
}
