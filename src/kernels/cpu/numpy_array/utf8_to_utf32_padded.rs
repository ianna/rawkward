// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Convert a UTF-8 flat array to padded UTF-32 codepoints.
//!
//! Corresponds to `src/cpu-kernels/awkward_NumpyArray_utf8_to_utf32_padded.cpp`.

use crate::kernels::cpu::error::KernelError;
use crate::kernels::cpu::unicode::utf8_codepoint_size;

// Masks for extracting payload bits (complement of the leading-byte masks).
const ONE_BYTE_PAYLOAD: u32 = 0b0111_1111;
const TWO_BYTE_PAYLOAD: u32 = 0b0001_1111;
const THREE_BYTE_PAYLOAD: u32 = 0b0000_1111;
const FOUR_BYTE_PAYLOAD: u32 = 0b0000_0111;
const CONTINUATION_PAYLOAD: u32 = 0b0011_1111;

/// Decode every UTF-8 sublist into padded UTF-32 codepoints.
///
/// For each sublist `k` in `fromoffsets[0..offsetslength-1]`, decode up to
/// `maxcodepoints` codepoints and zero-pad the rest.
///
/// # Errors
///
/// Returns an error if an invalid leading byte is encountered.
pub fn numpy_array_utf8_to_utf32_padded<C>(
    fromptr: &[u8],
    fromoffsets: &[C],
    offsetslength: usize,
    maxcodepoints: usize,
    toptr: &mut [u32],
) -> Result<(), KernelError>
where
    C: Copy + Into<i64>,
{
    let mut n_code_point = 0usize;
    let mut i_code_unit: usize = fromoffsets[0].into() as usize;

    for k in 0..offsetslength.saturating_sub(1) {
        let n_code_units = (fromoffsets[k + 1].into() - fromoffsets[k].into()) as usize;
        let end = i_code_unit + n_code_units;
        let mut n_sublist = 0usize;

        while i_code_unit < end {
            let b0 = fromptr[i_code_unit];
            let w = utf8_codepoint_size(b0);
            let cp = match w {
                1 => (b0 as u32) & ONE_BYTE_PAYLOAD,
                2 => {
                    ((b0 as u32) & TWO_BYTE_PAYLOAD) << 6
                        | (fromptr[i_code_unit + 1] as u32) & CONTINUATION_PAYLOAD
                }
                3 => {
                    ((b0 as u32) & THREE_BYTE_PAYLOAD) << 12
                        | ((fromptr[i_code_unit + 1] as u32) & CONTINUATION_PAYLOAD) << 6
                        | (fromptr[i_code_unit + 2] as u32) & CONTINUATION_PAYLOAD
                }
                4 => {
                    ((b0 as u32) & FOUR_BYTE_PAYLOAD) << 18
                        | ((fromptr[i_code_unit + 1] as u32) & CONTINUATION_PAYLOAD) << 12
                        | ((fromptr[i_code_unit + 2] as u32) & CONTINUATION_PAYLOAD) << 6
                        | (fromptr[i_code_unit + 3] as u32) & CONTINUATION_PAYLOAD
                }
                _ => {
                    return Err(KernelError::new(
                        "could not convert UTF8 code point to UTF32: invalid byte in UTF8 string",
                        k as i64,
                        b0 as i64,
                    ));
                }
            };
            toptr[n_code_point] = cp;
            n_code_point += 1;
            i_code_unit += w;
            n_sublist += 1;
        }
        // Zero-pad remaining slots.
        for _ in n_sublist..maxcodepoints {
            toptr[n_code_point] = 0;
            n_code_point += 1;
        }
    }
    Ok(())
}

pub fn numpy_array_utf8_to_utf32_padded_int32(
    fromptr: &[u8],
    fromoffsets: &[i32],
    offsetslength: usize,
    maxcodepoints: usize,
    toptr: &mut [u32],
) -> Result<(), KernelError> {
    numpy_array_utf8_to_utf32_padded(fromptr, fromoffsets, offsetslength, maxcodepoints, toptr)
}
pub fn numpy_array_utf8_to_utf32_padded_uint32(
    fromptr: &[u8],
    fromoffsets: &[u32],
    offsetslength: usize,
    maxcodepoints: usize,
    toptr: &mut [u32],
) -> Result<(), KernelError> {
    numpy_array_utf8_to_utf32_padded(fromptr, fromoffsets, offsetslength, maxcodepoints, toptr)
}
pub fn numpy_array_utf8_to_utf32_padded_int64(
    fromptr: &[u8],
    fromoffsets: &[i64],
    offsetslength: usize,
    maxcodepoints: usize,
    toptr: &mut [u32],
) -> Result<(), KernelError> {
    numpy_array_utf8_to_utf32_padded(fromptr, fromoffsets, offsetslength, maxcodepoints, toptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii() {
        // "hi" → [0x68, 0x69]
        let data = b"hi";
        let offsets = [0i64, 2];
        let mut out = [0u32; 3]; // maxcodepoints=3, pad 1
        numpy_array_utf8_to_utf32_padded_int64(data, &offsets, 2, 3, &mut out).unwrap();
        assert_eq!(out, [0x68, 0x69, 0]);
    }

    #[test]
    fn two_byte_char() {
        // 'é' = U+00E9 → UTF-8 [0xC3, 0xA9]
        let data = [0xC3u8, 0xA9];
        let offsets = [0i64, 2];
        let mut out = [0u32; 1];
        numpy_array_utf8_to_utf32_padded_int64(&data, &offsets, 2, 1, &mut out).unwrap();
        assert_eq!(out[0], 0x00E9);
    }

    #[test]
    fn invalid_byte_error() {
        let data = [0x80u8]; // continuation byte as leader
        let offsets = [0i64, 1];
        let mut out = [0u32; 1];
        assert!(numpy_array_utf8_to_utf32_padded_int64(&data, &offsets, 2, 1, &mut out).is_err());
    }
}
