// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! UTF-8 helper utilities.
//!
//! Corresponds to `src/cpu-kernels/unicode.cpp`.

// Masks and bit-patterns for each UTF-8 sequence length.
const ONE_BYTE_MASK: u8 = 0b1000_0000;
const ONE_BYTE_BITS: u8 = 0b0000_0000; // 0xxxxxxx

const TWO_BYTES_MASK: u8 = 0b1110_0000;
const TWO_BYTES_BITS: u8 = 0b1100_0000; // 110xxxxx

const THREE_BYTES_MASK: u8 = 0b1111_0000;
const THREE_BYTES_BITS: u8 = 0b1110_0000; // 1110xxxx

const FOUR_BYTES_MASK: u8 = 0b1111_1000;
const FOUR_BYTES_BITS: u8 = 0b1111_0000; // 11110xxx

/// Return the number of bytes in the UTF-8 codepoint whose first byte is
/// `byte`, or `0` if `byte` is not a valid leading byte.
///
/// # Examples
///
/// ```
/// use cpu_kernels::unicode::utf8_codepoint_size;
///
/// assert_eq!(utf8_codepoint_size(b'A'), 1);           // ASCII
/// assert_eq!(utf8_codepoint_size(0xC3), 2);           // e.g. 'é'
/// assert_eq!(utf8_codepoint_size(0xE2), 3);           // e.g. '€'
/// assert_eq!(utf8_codepoint_size(0xF0), 4);           // e.g. '𐍈'
/// assert_eq!(utf8_codepoint_size(0x80), 0);           // continuation byte – invalid
/// ```
#[inline]
pub fn utf8_codepoint_size(byte: u8) -> usize {
    if (byte & ONE_BYTE_MASK) == ONE_BYTE_BITS {
        1
    } else if (byte & TWO_BYTES_MASK) == TWO_BYTES_BITS {
        2
    } else if (byte & THREE_BYTES_MASK) == THREE_BYTES_BITS {
        3
    } else if (byte & FOUR_BYTES_MASK) == FOUR_BYTES_BITS {
        4
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_is_one_byte() {
        for b in 0u8..=127 {
            assert_eq!(utf8_codepoint_size(b), 1, "byte 0x{b:02x}");
        }
    }

    #[test]
    fn two_byte_leaders() {
        // 0xC0..=0xDF are all two-byte leaders.
        for b in 0xC0u8..=0xDF {
            assert_eq!(utf8_codepoint_size(b), 2, "byte 0x{b:02x}");
        }
    }

    #[test]
    fn three_byte_leaders() {
        for b in 0xE0u8..=0xEF {
            assert_eq!(utf8_codepoint_size(b), 3, "byte 0x{b:02x}");
        }
    }

    #[test]
    fn four_byte_leaders() {
        for b in 0xF0u8..=0xF7 {
            assert_eq!(utf8_codepoint_size(b), 4, "byte 0x{b:02x}");
        }
    }

    #[test]
    fn continuation_bytes_are_invalid() {
        // 0x80..=0xBF are continuation bytes.
        for b in 0x80u8..=0xBF {
            assert_eq!(utf8_codepoint_size(b), 0, "byte 0x{b:02x}");
        }
    }

    #[test]
    fn specific_known_leaders() {
        assert_eq!(utf8_codepoint_size(b'A'), 1); // U+0041
        assert_eq!(utf8_codepoint_size(0xC3), 2); // 'é' first byte
        assert_eq!(utf8_codepoint_size(0xE2), 3); // '€' first byte
        assert_eq!(utf8_codepoint_size(0xF0), 4); // '𐍈' first byte
    }
}
