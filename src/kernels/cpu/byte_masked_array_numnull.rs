// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Count the number of null (invalid) entries in a byte-masked array.
//!
//! Corresponds to `src/cpu-kernels/awkward_ByteMaskedArray_numnull.cpp`.

/// Count the number of null (invalid) entries in `mask`.
///
/// An entry is **null** when `(mask[i] != 0) != validwhen`, i.e. the opposite
/// of "valid".
///
/// # Returns
///
/// The count of null entries as `i64`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::byte_masked_array_numnull::byte_masked_array_numnull;
///
/// let mask = [1i8, 0, 1, 0, 0]; // validwhen=true → indices 1,3,4 are null
/// assert_eq!(byte_masked_array_numnull(&mask, true), 3);
/// ```
pub fn byte_masked_array_numnull(mask: &[i8], validwhen: bool) -> i64 {
    mask.iter().filter(|&&m| (m != 0) != validwhen).count() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_valid() {
        let mask = [1i8; 5];
        assert_eq!(byte_masked_array_numnull(&mask, true), 0);
    }

    #[test]
    fn all_null() {
        let mask = [0i8; 5];
        assert_eq!(byte_masked_array_numnull(&mask, true), 5);
    }

    #[test]
    fn mixed() {
        let mask = [1i8, 0, 1, 0, 0];
        assert_eq!(byte_masked_array_numnull(&mask, true), 3);
    }

    #[test]
    fn validwhen_false() {
        // With validwhen=false: 0 = valid, non-zero = null
        let mask = [1i8, 0, 0, 1];
        assert_eq!(byte_masked_array_numnull(&mask, false), 2);
    }

    #[test]
    fn empty_mask() {
        assert_eq!(byte_masked_array_numnull(&[], true), 0);
    }
}
