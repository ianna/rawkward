// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Convert a bit-packed mask to a byte mask.
//!
//! Corresponds to `src/cpu-kernels/awkward_BitMaskedArray_to_ByteMaskedArray.cpp`.

/// Convert a bit-packed boolean mask (`frombitmask`) into a flat byte mask
/// (`tobytemask`).
///
/// Each bit in `frombitmask` becomes one `i8` byte in `tobytemask`. A byte is
/// **`0`** (valid / not-null) when the bit equals `validwhen`, and **`1`**
/// (invalid / null) otherwise.
///
/// # Parameters
///
/// * `tobytemask`   – Output slice; must have length `frombitmask.len() * 8`.
/// * `frombitmask`  – Input bit-packed mask.
/// * `validwhen`    – The bit value (`true` = 1, `false` = 0) that means
///                    "this element is valid".
/// * `lsb_order`    – When `true` the least-significant bit of each byte
///                    corresponds to the first element.  When `false` the
///                    most-significant bit does.
///
/// # Panics
///
/// Panics if `tobytemask.len() < frombitmask.len() * 8`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::bit_masked_array_to_byte_masked_array::bit_masked_array_to_byte_masked_array;
///
/// let bitmask = [0b0000_0101u8]; // bits 0 and 2 set (LSB order)
/// let mut bytemask = [0i8; 8];
/// bit_masked_array_to_byte_masked_array(&mut bytemask, &bitmask, true, true);
/// // bit 0 = 1 == validwhen(true) → byte 0 = 0 (valid)
/// // bit 1 = 0 != validwhen(true) → byte 1 = 1 (null)
/// assert_eq!(bytemask[0], 0);
/// assert_eq!(bytemask[1], 1);
/// assert_eq!(bytemask[2], 0);
/// ```
pub fn bit_masked_array_to_byte_masked_array(
    tobytemask: &mut [i8],
    frombitmask: &[u8],
    validwhen: bool,
    lsb_order: bool,
) {
    assert!(
        tobytemask.len() >= frombitmask.len() * 8,
        "tobytemask must hold frombitmask.len() * 8 elements"
    );

    if lsb_order {
        for (i, &packed) in frombitmask.iter().enumerate() {
            let mut byte = packed;
            for bit in 0..8usize {
                tobytemask[i * 8 + bit] = i8::from((byte & 1 != 0) != validwhen);
                byte >>= 1;
            }
        }
    } else {
        for (i, &packed) in frombitmask.iter().enumerate() {
            let mut byte = packed;
            for bit in 0..8usize {
                tobytemask[i * 8 + bit] = i8::from(((byte & 0x80) != 0) != validwhen);
                byte <<= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsb_all_valid() {
        // All bits set, validwhen=true → all bytes 0
        let bitmask = [0xFFu8];
        let mut bytemask = [0i8; 8];
        bit_masked_array_to_byte_masked_array(&mut bytemask, &bitmask, true, true);
        assert!(bytemask.iter().all(|&b| b == 0));
    }

    #[test]
    fn lsb_all_null() {
        // No bits set, validwhen=true → all bytes 1
        let bitmask = [0x00u8];
        let mut bytemask = [0i8; 8];
        bit_masked_array_to_byte_masked_array(&mut bytemask, &bitmask, true, true);
        assert!(bytemask.iter().all(|&b| b == 1));
    }

    #[test]
    fn lsb_alternating() {
        let bitmask = [0b1010_1010u8];
        let mut bytemask = [0i8; 8];
        bit_masked_array_to_byte_masked_array(&mut bytemask, &bitmask, true, true);
        // bits: 0=0,1=1,2=0,3=1,4=0,5=1,6=0,7=1
        let expected: [i8; 8] = [1, 0, 1, 0, 1, 0, 1, 0];
        assert_eq!(bytemask, expected);
    }

    #[test]
    fn msb_order() {
        // 0b1000_0000 → only bit 7 (MSB) set
        let bitmask = [0b1000_0000u8];
        let mut bytemask = [0i8; 8];
        bit_masked_array_to_byte_masked_array(&mut bytemask, &bitmask, true, false);
        // In MSB order byte 0 corresponds to bit 7 (the MSB)
        assert_eq!(bytemask[0], 0); // valid
        for &b in &bytemask[1..] {
            assert_eq!(b, 1); // null
        }
    }

    #[test]
    fn multi_byte_bitmask() {
        let bitmask = [0xFF, 0x00];
        let mut bytemask = [0i8; 16];
        bit_masked_array_to_byte_masked_array(&mut bytemask, &bitmask, true, true);
        assert!(bytemask[..8].iter().all(|&b| b == 0));
        assert!(bytemask[8..].iter().all(|&b| b == 1));
    }

    #[test]
    fn validwhen_false_inverts() {
        let bitmask = [0xFFu8];
        let mut bytemask = [0i8; 8];
        bit_masked_array_to_byte_masked_array(&mut bytemask, &bitmask, false, true);
        // bit set but validwhen=false → all null
        assert!(bytemask.iter().all(|&b| b == 1));
    }
}
