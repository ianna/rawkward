// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Convert a byte mask to an indexed option array.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ByteMaskedArray_toIndexedOptionArray.cpp`.

/// Convert a byte mask into an `IndexedOptionArray` index.
///
/// For each position `i`:
/// * If `(mask[i] != 0) == validwhen`, write `i` (the element is valid and
///   lives at position `i` in the content).
/// * Otherwise write `-1` (the element is null).
///
/// # Parameters
///
/// * `toindex`   – Output index; same length as `mask`.
/// * `mask`      – Input byte-mask array.
/// * `validwhen` – The byte truth-value that signals "valid".
///
/// # Panics
///
/// Panics if `toindex.len() != mask.len()`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::byte_masked_array_to_indexed_option_array::byte_masked_array_to_indexed_option_array_64;
///
/// let mask = [1i8, 0, 1, 0];
/// let mut index = [0i64; 4];
/// byte_masked_array_to_indexed_option_array_64(&mut index, &mask, true);
/// assert_eq!(index, [0, -1, 2, -1]);
/// ```
pub fn byte_masked_array_to_indexed_option_array<T>(toindex: &mut [T], mask: &[i8], validwhen: bool)
where
    T: TryFrom<i64> + Copy,
    <T as TryFrom<i64>>::Error: std::fmt::Debug,
{
    assert_eq!(toindex.len(), mask.len());
    let minus_one = T::try_from(-1i64).expect("-1 fits in T");
    for (i, &m) in mask.iter().enumerate() {
        toindex[i] = if (m != 0) == validwhen {
            T::try_from(i as i64).expect("index fits in T")
        } else {
            minus_one
        };
    }
}

/// Typed wrapper for `i64` (mirrors `awkward_ByteMaskedArray_toIndexedOptionArray64`).
pub fn byte_masked_array_to_indexed_option_array_64(
    toindex: &mut [i64],
    mask: &[i8],
    validwhen: bool,
) {
    byte_masked_array_to_indexed_option_array(toindex, mask, validwhen);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_valid() {
        let mask = [1i8; 4];
        let mut index = [0i64; 4];
        byte_masked_array_to_indexed_option_array_64(&mut index, &mask, true);
        assert_eq!(index, [0, 1, 2, 3]);
    }

    #[test]
    fn all_null() {
        let mask = [0i8; 4];
        let mut index = [0i64; 4];
        byte_masked_array_to_indexed_option_array_64(&mut index, &mask, true);
        assert_eq!(index, [-1, -1, -1, -1]);
    }

    #[test]
    fn alternating() {
        let mask = [1i8, 0, 1, 0];
        let mut index = [0i64; 4];
        byte_masked_array_to_indexed_option_array_64(&mut index, &mask, true);
        assert_eq!(index, [0, -1, 2, -1]);
    }

    #[test]
    fn validwhen_false() {
        let mask = [0i8, 1, 0, 1];
        let mut index = [0i64; 4];
        byte_masked_array_to_indexed_option_array_64(&mut index, &mask, false);
        assert_eq!(index, [0, -1, 2, -1]);
    }
}
