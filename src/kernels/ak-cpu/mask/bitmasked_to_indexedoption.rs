// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Convert a bit-packed mask to an indexed option array.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_BitMaskedArray_to_IndexedOptionArray.cpp`.

/// Convert a bit-packed boolean mask into an index array for an
/// `IndexedOptionArray`.
///
/// For each bit position `p` (0-based):
/// * If the bit equals `validwhen`, write `p` into `toindex[p]` (the element
///   is valid and its content is at position `p`).
/// * Otherwise write `-1` into `toindex[p]` (the element is null / missing).
///
/// # Parameters
///
/// * `toindex`     – Output slice; length must be `frombitmask.len() * 8`.
/// * `frombitmask` – Input bit-packed mask.
/// * `validwhen`   – The bit value that means "valid".
/// * `lsb_order`   – `true` = LSB-first; `false` = MSB-first.
///
/// # Panics
///
/// Panics if `toindex.len() < frombitmask.len() * 8`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::bit_masked_array_to_indexed_option_array::bit_masked_array_to_indexed_option_array_64;
///
/// // Single byte: bits 0 and 2 are set (LSB order, validwhen=true)
/// let bitmask = [0b0000_0101u8];
/// let mut index = [0i64; 8];
/// bit_masked_array_to_indexed_option_array_64(&mut index, &bitmask, true, true);
/// assert_eq!(index[0], 0);  // bit 0 set → valid, points to index 0
/// assert_eq!(index[1], -1); // bit 1 not set → null
/// assert_eq!(index[2], 2);  // bit 2 set → valid, points to index 2
/// ```
pub fn bit_masked_array_to_indexed_option_array<T>(
    toindex: &mut [T],
    frombitmask: &[u8],
    validwhen: bool,
    lsb_order: bool,
) where
    T: TryFrom<i64> + Copy,
    <T as TryFrom<i64>>::Error: std::fmt::Debug,
{
    assert!(
        toindex.len() >= frombitmask.len() * 8,
        "toindex must hold frombitmask.len() * 8 elements"
    );

    let encode = |valid: bool, pos: i64| -> T {
        if valid {
            T::try_from(pos).expect("index fits in T")
        } else {
            T::try_from(-1i64).expect("-1 fits in T")
        }
    };

    if lsb_order {
        for (i, &orig) in frombitmask.iter().enumerate() {
            let mut byte = orig;
            for bit in 0..8usize {
                let valid = (byte & 1 != 0) == validwhen;
                toindex[i * 8 + bit] = encode(valid, (i * 8 + bit) as i64);
                byte >>= 1;
            }
        }
    } else {
        for (i, &orig) in frombitmask.iter().enumerate() {
            let mut byte = orig;
            for bit in 0..8usize {
                let valid = ((byte & 0x80) != 0) == validwhen;
                toindex[i * 8 + bit] = encode(valid, (i * 8 + bit) as i64);
                byte <<= 1;
            }
        }
    }
}

/// Typed wrapper for `i64` output (mirrors `awkward_BitMaskedArray_to_IndexedOptionArray64`).
pub fn bit_masked_array_to_indexed_option_array_64(
    toindex: &mut [i64],
    frombitmask: &[u8],
    validwhen: bool,
    lsb_order: bool,
) {
    bit_masked_array_to_indexed_option_array(toindex, frombitmask, validwhen, lsb_order);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsb_all_valid() {
        let bitmask = [0xFFu8];
        let mut index = [0i64; 8];
        bit_masked_array_to_indexed_option_array_64(&mut index, &bitmask, true, true);
        let expected: [i64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(index, expected);
    }

    #[test]
    fn lsb_all_null() {
        let bitmask = [0x00u8];
        let mut index = [0i64; 8];
        bit_masked_array_to_indexed_option_array_64(&mut index, &bitmask, true, true);
        assert!(index.iter().all(|&v| v == -1));
    }

    #[test]
    fn lsb_alternating() {
        // 0b0101_0101 → bits 0,2,4,6 set
        let bitmask = [0b0101_0101u8];
        let mut index = [0i64; 8];
        bit_masked_array_to_indexed_option_array_64(&mut index, &bitmask, true, true);
        let expected: [i64; 8] = [0, -1, 2, -1, 4, -1, 6, -1];
        assert_eq!(index, expected);
    }

    #[test]
    fn msb_order() {
        // 0b1000_0000 → only MSB (bit 7) set → position 0 in MSB order
        let bitmask = [0b1000_0000u8];
        let mut index = [0i64; 8];
        bit_masked_array_to_indexed_option_array_64(&mut index, &bitmask, true, false);
        assert_eq!(index[0], 0);
        for &v in &index[1..] {
            assert_eq!(v, -1);
        }
    }

    #[test]
    fn multi_byte() {
        let bitmask = [0xFF, 0x00];
        let mut index = [0i64; 16];
        bit_masked_array_to_indexed_option_array_64(&mut index, &bitmask, true, true);
        for i in 0..8 {
            assert_eq!(index[i], i as i64);
        }
        for v in &index[8..] {
            assert_eq!(*v, -1);
        }
    }
}
