// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! IndexedArray fill kernels
//!
//! Port of Awkward's `awkward_IndexedArray_fill` kernels.
//!
//! Semantics:
//! - Negative values map to -1 (null sentinel)
//! - Non-negative values are shifted by `base`
//! - Written into output with an offset

/// Fill from i32 → i64
pub fn indexed_array_fill_to64_from32(
    toindex: &mut [i64],
    toindex_offset: usize,
    fromindex: &[i32],
    base: i64,
) {
    let len = fromindex.len();
    assert!(toindex_offset + len <= toindex.len());

    for i in 0..len {
        let fromval = fromindex[i];

        toindex[toindex_offset + i] = if fromval < 0 {
            -1
        } else {
            fromval as i64 + base
        };
    }
}

/// Fill from i64 → i64
pub fn indexed_array_fill_to64_from64(
    toindex: &mut [i64],
    toindex_offset: usize,
    fromindex: &[i64],
    base: i64,
) {
    let len = fromindex.len();
    assert!(toindex_offset + len <= toindex.len());

    for i in 0..len {
        let fromval = fromindex[i];

        toindex[toindex_offset + i] = if fromval < 0 { -1 } else { fromval + base };
    }
}

/// Fill from u32 → i64
///
/// Note: unsigned values cannot be negative,
/// so no null sentinel check is needed.
pub fn indexed_array_fill_to64_from_u32(
    toindex: &mut [i64],
    toindex_offset: usize,
    fromindex: &[u32],
    base: i64,
) {
    let len = fromindex.len();
    assert!(toindex_offset + len <= toindex.len());

    for i in 0..len {
        let fromval = fromindex[i];
        toindex[toindex_offset + i] = fromval as i64 + base;
    }
}

//
// Optional: unsafe fast versions (closer to C++)
//

/// Unsafe optimized version (i32 → i64)
pub fn indexed_array_fill_to64_from32_unchecked(
    toindex: &mut [i64],
    toindex_offset: usize,
    fromindex: &[i32],
    base: i64,
) {
    let len = fromindex.len();
    assert!(toindex_offset + len <= toindex.len());

    unsafe {
        for i in 0..len {
            let fromval = *fromindex.get_unchecked(i);

            *toindex.get_unchecked_mut(toindex_offset + i) = if fromval < 0 {
                -1
            } else {
                fromval as i64 + base
            };
        }
    }
}

//
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_i32_basic() {
        let from = vec![1, 2, -1, 4];
        let mut to = vec![0; 4];

        indexed_array_fill_to64_from32(&mut to, 0, &from, 10);

        assert_eq!(to, vec![11, 12, -1, 14]);
    }

    #[test]
    fn test_fill_i64_basic() {
        let from = vec![1_i64, -1, 3];
        let mut to = vec![0; 3];

        indexed_array_fill_to64_from64(&mut to, 0, &from, 5);

        assert_eq!(to, vec![6, -1, 8]);
    }

    #[test]
    fn test_fill_u32_basic() {
        let from = vec![1_u32, 2, 3];
        let mut to = vec![0; 3];

        indexed_array_fill_to64_from_u32(&mut to, 0, &from, 7);

        assert_eq!(to, vec![8, 9, 10]);
    }

    #[test]
    fn test_with_offset() {
        let from = vec![1, 2];
        let mut to = vec![0; 5];

        indexed_array_fill_to64_from32(&mut to, 2, &from, 10);

        assert_eq!(to, vec![0, 0, 11, 12, 0]);
    }
}
