// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build a ByteMaskedArray mask from parent indices.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_NumpyArray_reduce_mask_ByteMaskedArray_64.cpp`.

/// Initialise `toptr` to all `1` (masked / null), then clear to `0` any group
/// that has at least one contributing element.
///
/// Groups present in `parents` will have `toptr[g] = 0` (valid); groups absent
/// will remain `1` (null).
pub fn numpy_array_reduce_mask_byte_masked_array_64(toptr: &mut [i8], parents: &[i64]) {
    for v in toptr.iter_mut() {
        *v = 1;
    }
    for &p in parents {
        toptr[p as usize] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_present_groups() {
        let parents = [0i64, 2];
        let mut out = [1i8; 4];
        numpy_array_reduce_mask_byte_masked_array_64(&mut out, &parents);
        assert_eq!(out, [0, 1, 0, 1]);
    }

    #[test]
    fn all_present() {
        let parents = [0i64, 1, 2];
        let mut out = [1i8; 3];
        numpy_array_reduce_mask_byte_masked_array_64(&mut out, &parents);
        assert_eq!(out, [0, 0, 0]);
    }

    #[test]
    fn empty_parents_all_masked() {
        let parents: [i64; 0] = [];
        let mut out = [1i8; 3];
        numpy_array_reduce_mask_byte_masked_array_64(&mut out, &parents);
        assert_eq!(out, [1, 1, 1]);
    }
}
