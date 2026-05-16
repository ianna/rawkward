// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Collect carry indices for valid entries in a byte-masked array.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ByteMaskedArray_getitem_nextcarry.cpp`.

/// Fill `tocarry` with the indices of every element in `mask` that is valid.
///
/// An element at position `i` is considered valid when
/// `(mask[i] != 0) == validwhen`.
///
/// # Parameters
///
/// * `tocarry`   – Output buffer.  Must be pre-allocated; only the first
///                 returned `usize` (the count of valid elements) entries are
///                 written.
/// * `mask`      – Byte-mask array (`0` = false, non-zero = true).
/// * `validwhen` – Which byte-truth value (`true` or `false`) marks a
///                 position as valid.
///
/// # Returns
///
/// The number of valid elements written into `tocarry`.
///
/// # Panics
///
/// Panics if `tocarry` is shorter than the number of valid elements.
///
/// # Examples
///
/// ```
/// use cpu_kernels::byte_masked_array_getitem_nextcarry::byte_masked_array_getitem_nextcarry_64;
///
/// let mask = [1i8, 0, 1, 0, 1];
/// let mut carry = [0i64; 5];
/// let n = byte_masked_array_getitem_nextcarry_64(&mut carry, &mask, true);
/// assert_eq!(n, 3);
/// assert_eq!(&carry[..n], &[0, 2, 4]);
/// ```
pub fn byte_masked_array_getitem_nextcarry<T>(
    tocarry: &mut [T],
    mask: &[i8],
    validwhen: bool,
) -> usize
where
    T: TryFrom<usize> + Copy,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut k = 0usize;
    for (i, &m) in mask.iter().enumerate() {
        if (m != 0) == validwhen {
            tocarry[k] = T::try_from(i).expect("index fits in T");
            k += 1;
        }
    }
    k
}

/// Typed wrapper for `i64` carry (mirrors
/// `awkward_ByteMaskedArray_getitem_nextcarry_64`).
pub fn byte_masked_array_getitem_nextcarry_64(
    tocarry: &mut [i64],
    mask: &[i8],
    validwhen: bool,
) -> usize {
    byte_masked_array_getitem_nextcarry(tocarry, mask, validwhen)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_valid() {
        let mask = [1i8; 4];
        let mut carry = [0i64; 4];
        let n = byte_masked_array_getitem_nextcarry_64(&mut carry, &mask, true);
        assert_eq!(n, 4);
        assert_eq!(&carry[..n], &[0, 1, 2, 3]);
    }

    #[test]
    fn all_null() {
        let mask = [0i8; 4];
        let mut carry = [0i64; 4];
        let n = byte_masked_array_getitem_nextcarry_64(&mut carry, &mask, true);
        assert_eq!(n, 0);
    }

    #[test]
    fn alternating() {
        let mask = [1i8, 0, 1, 0, 1];
        let mut carry = [0i64; 5];
        let n = byte_masked_array_getitem_nextcarry_64(&mut carry, &mask, true);
        assert_eq!(n, 3);
        assert_eq!(&carry[..n], &[0, 2, 4]);
    }

    #[test]
    fn validwhen_false() {
        // When validwhen=false the zeros in the mask are the valid positions.
        let mask = [1i8, 0, 1, 0];
        let mut carry = [0i64; 4];
        let n = byte_masked_array_getitem_nextcarry_64(&mut carry, &mask, false);
        assert_eq!(n, 2);
        assert_eq!(&carry[..n], &[1, 3]);
    }

    #[test]
    fn nonzero_values_treated_as_true() {
        let mask = [2i8, -1, 0]; // 2 and -1 are both non-zero
        let mut carry = [0i64; 3];
        let n = byte_masked_array_getitem_nextcarry_64(&mut carry, &mask, true);
        assert_eq!(n, 2);
        assert_eq!(&carry[..n], &[0, 1]);
    }
}
