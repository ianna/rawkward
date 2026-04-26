// Copyright (c) 2026 Ianna Osborne 
// SPDX-License-Identifier: BSD-3-Clause

//! Compute the maximum codepoint count across all UTF-8 sublists.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_NumpyArray_prepare_utf8_to_utf32_padded.cpp`.

use crate::kernels::cpu::unicode::utf8_codepoint_size;

/// Scan all sublists in `fromptr` and return the maximum number of Unicode
/// codepoints in any single sublist.  This is the `maxcodepoints` value needed
/// by `numpy_array_utf8_to_utf32_padded`.
pub fn numpy_array_prepare_utf8_to_utf32_padded<C>(
    fromptr: &[u8],
    fromoffsets: &[C],
    offsetslength: usize,
) -> i64
where
    C: Copy + Into<i64>,
{
    let mut maxcodepoints = 0i64;
    let mut i_code_unit: usize = fromoffsets[0].into() as usize;

    for k in 0..offsetslength.saturating_sub(1) {
        let n_code_units = (fromoffsets[k + 1].into() - fromoffsets[k].into()) as usize;
        let end = i_code_unit + n_code_units;
        let mut n_sublist = 0i64;

        while i_code_unit < end {
            let w = utf8_codepoint_size(fromptr[i_code_unit]);
            let w = if w == 0 { 1 } else { w }; // skip invalid bytes gracefully
            i_code_unit += w;
            n_sublist += 1;
        }
        if n_sublist > maxcodepoints {
            maxcodepoints = n_sublist;
        }
    }
    maxcodepoints
}

pub fn numpy_array_prepare_utf8_to_utf32_padded_int32(fromptr: &[u8], fromoffsets: &[i32], offsetslength: usize) -> i64 {
    numpy_array_prepare_utf8_to_utf32_padded(fromptr, fromoffsets, offsetslength)
}
pub fn numpy_array_prepare_utf8_to_utf32_padded_uint32(fromptr: &[u8], fromoffsets: &[u32], offsetslength: usize) -> i64 {
    numpy_array_prepare_utf8_to_utf32_padded(fromptr, fromoffsets, offsetslength)
}
pub fn numpy_array_prepare_utf8_to_utf32_padded_int64(fromptr: &[u8], fromoffsets: &[i64], offsetslength: usize) -> i64 {
    numpy_array_prepare_utf8_to_utf32_padded(fromptr, fromoffsets, offsetslength)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_strings() {
        // "hi" (2 chars), "hello" (5 chars) → max=5
        let data    = b"hihello";
        let offsets = [0i64, 2, 7];
        assert_eq!(numpy_array_prepare_utf8_to_utf32_padded_int64(data, &offsets, 3), 5);
    }

    #[test]
    fn multibyte() {
        // 'é' is 2 bytes but 1 codepoint
        let data    = [0xC3u8, 0xA9, b'a']; // "é", "a"
        let offsets = [0i64, 2, 3];
        assert_eq!(numpy_array_prepare_utf8_to_utf32_padded_int64(&data, &offsets, 3), 1);
    }
}
