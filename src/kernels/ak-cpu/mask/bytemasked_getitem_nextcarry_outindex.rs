// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Collect carry indices and build the out-index for a byte-masked array.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ByteMaskedArray_getitem_nextcarry_outindex.cpp`.

/// Simultaneously fill `tocarry` with the positions of valid elements and
/// `outindex` with a forward-mapping from original positions to compact
/// positions (`-1` for null entries).
///
/// For each position `i`:
/// * If `(mask[i] != 0) == validwhen`:
///   - `tocarry[k] = i`
///   - `outindex[i] = k`
///   - `k` incremented.
/// * Otherwise: `outindex[i] = -1`.
///
/// # Parameters
///
/// * `tocarry`   – Output carry buffer; must be large enough for all valid
///                 elements.
/// * `outindex`  – Output index; same length as `mask`.
/// * `mask`      – Byte-mask array.
/// * `validwhen` – Byte truth-value that signals "valid".
///
/// # Returns
///
/// The number of valid elements (`k` after the loop).
///
/// # Examples
///
/// ```
/// use cpu_kernels::byte_masked_array_getitem_nextcarry_outindex::byte_masked_array_getitem_nextcarry_outindex_64;
///
/// let mask     = [1i8, 0, 1, 0];
/// let mut carry   = [0i64; 4];
/// let mut outidx  = [0i64; 4];
/// let n = byte_masked_array_getitem_nextcarry_outindex_64(&mut carry, &mut outidx, &mask, true);
/// assert_eq!(n, 2);
/// assert_eq!(&carry[..n],  &[0, 2]);
/// assert_eq!(outidx, [0, -1, 1, -1]);
/// ```
pub fn byte_masked_array_getitem_nextcarry_outindex<T>(
    tocarry: &mut [T],
    outindex: &mut [T],
    mask: &[i8],
    validwhen: bool,
) -> usize
where
    T: TryFrom<i64> + Copy,
    <T as TryFrom<i64>>::Error: std::fmt::Debug,
{
    let minus_one = T::try_from(-1i64).expect("-1 fits in T");
    let mut k = 0i64;
    for (i, &m) in mask.iter().enumerate() {
        if (m != 0) == validwhen {
            tocarry[k as usize] = T::try_from(i as i64).expect("index fits in T");
            outindex[i] = T::try_from(k).expect("k fits in T");
            k += 1;
        } else {
            outindex[i] = minus_one;
        }
    }
    k as usize
}

/// Typed wrapper for `i64` (mirrors
/// `awkward_ByteMaskedArray_getitem_nextcarry_outindex_64`).
pub fn byte_masked_array_getitem_nextcarry_outindex_64(
    tocarry: &mut [i64],
    outindex: &mut [i64],
    mask: &[i8],
    validwhen: bool,
) -> usize {
    byte_masked_array_getitem_nextcarry_outindex(tocarry, outindex, mask, validwhen)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_valid() {
        let mask = [1i8; 4];
        let mut carry = [0i64; 4];
        let mut outidx = [0i64; 4];
        let n =
            byte_masked_array_getitem_nextcarry_outindex_64(&mut carry, &mut outidx, &mask, true);
        assert_eq!(n, 4);
        assert_eq!(&carry[..n], &[0, 1, 2, 3]);
        assert_eq!(outidx, [0, 1, 2, 3]);
    }

    #[test]
    fn all_null() {
        let mask = [0i8; 4];
        let mut carry = [0i64; 4];
        let mut outidx = [0i64; 4];
        let n =
            byte_masked_array_getitem_nextcarry_outindex_64(&mut carry, &mut outidx, &mask, true);
        assert_eq!(n, 0);
        assert_eq!(outidx, [-1, -1, -1, -1]);
    }

    #[test]
    fn mixed() {
        let mask = [1i8, 0, 1, 0];
        let mut carry = [0i64; 4];
        let mut outidx = [0i64; 4];
        let n =
            byte_masked_array_getitem_nextcarry_outindex_64(&mut carry, &mut outidx, &mask, true);
        assert_eq!(n, 2);
        assert_eq!(&carry[..n], &[0, 2]);
        assert_eq!(outidx, [0, -1, 1, -1]);
    }

    #[test]
    fn validwhen_false() {
        let mask = [0i8, 1, 0, 1];
        let mut carry = [0i64; 4];
        let mut outidx = [0i64; 4];
        let n =
            byte_masked_array_getitem_nextcarry_outindex_64(&mut carry, &mut outidx, &mask, false);
        assert_eq!(n, 2);
        assert_eq!(&carry[..n], &[0, 2]);
        assert_eq!(outidx, [0, -1, 1, -1]);
    }
}
