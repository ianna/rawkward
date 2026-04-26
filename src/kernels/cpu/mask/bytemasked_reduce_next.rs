// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build the carry, parent, and out-index arrays for the next reduction step
//! of a byte-masked array.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ByteMaskedArray_reduce_next_64.cpp`.

/// Simultaneously populate `nextcarry`, `nextparents`, and `outindex` for the
/// next reduction step over a byte-masked array.
///
/// For each position `i` in `mask`:
/// * **Valid** (`(mask[i] != 0) == validwhen`):
///   - `nextcarry[k] = i`
///   - `nextparents[k] = parents[i]`
///   - `outindex[i] = k`
///   - `k` incremented.
/// * **Null**:
///   - `outindex[i] = -1`.
///
/// # Parameters
///
/// * `nextcarry`    – Output carry buffer; must hold all valid entries.
/// * `nextparents`  – Output parents for carry entries; same length as `nextcarry`.
/// * `outindex`     – Output index; same length as `mask`.
/// * `mask`         – Byte-mask array.
/// * `parents`      – Parent-group indices; same length as `mask`.
/// * `validwhen`    – The byte truth-value that means "valid".
///
/// # Returns
///
/// The number of valid entries written (`k`).
///
/// # Panics
///
/// Panics if `mask.len() != parents.len()` or if output buffers are too short.
///
/// # Examples
///
/// ```
/// use cpu_kernels::byte_masked_array_reduce_next_64::byte_masked_array_reduce_next_64;
///
/// let mask    = [1i8, 0, 1];
/// let parents = [0i64, 0, 1];
/// let mut nextcarry   = [0i64; 3];
/// let mut nextparents = [0i64; 3];
/// let mut outindex    = [0i64; 3];
/// let n = byte_masked_array_reduce_next_64(
///     &mut nextcarry, &mut nextparents, &mut outindex, &mask, &parents, true,
/// );
/// assert_eq!(n, 2);
/// assert_eq!(&nextcarry[..n],   &[0, 2]);
/// assert_eq!(&nextparents[..n], &[0, 1]);
/// assert_eq!(outindex,           [0, -1, 1]);
/// ```
pub fn byte_masked_array_reduce_next_64(
    nextcarry: &mut [i64],
    nextparents: &mut [i64],
    outindex: &mut [i64],
    mask: &[i8],
    parents: &[i64],
    validwhen: bool,
) -> usize {
    assert_eq!(mask.len(), parents.len());
    assert_eq!(mask.len(), outindex.len());

    let mut k = 0usize;
    for (i, (&m, &p)) in mask.iter().zip(parents.iter()).enumerate() {
        if (m != 0) == validwhen {
            nextcarry[k] = i as i64;
            nextparents[k] = p;
            outindex[i] = k as i64;
            k += 1;
        } else {
            outindex[i] = -1;
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_valid() {
        let mask = [1i8; 4];
        let parents = [0i64, 0, 1, 1];
        let mut carry = [0i64; 4];
        let mut parents_out = [0i64; 4];
        let mut outidx = [0i64; 4];
        let n = byte_masked_array_reduce_next_64(
            &mut carry,
            &mut parents_out,
            &mut outidx,
            &mask,
            &parents,
            true,
        );
        assert_eq!(n, 4);
        assert_eq!(&carry[..n], &[0, 1, 2, 3]);
        assert_eq!(&parents_out[..n], &[0, 0, 1, 1]);
        assert_eq!(outidx, [0, 1, 2, 3]);
    }

    #[test]
    fn all_null() {
        let mask = [0i8; 4];
        let parents = [0i64; 4];
        let mut carry = [0i64; 4];
        let mut parents_out = [0i64; 4];
        let mut outidx = [0i64; 4];
        let n = byte_masked_array_reduce_next_64(
            &mut carry,
            &mut parents_out,
            &mut outidx,
            &mask,
            &parents,
            true,
        );
        assert_eq!(n, 0);
        assert_eq!(outidx, [-1, -1, -1, -1]);
    }

    #[test]
    fn mixed() {
        let mask = [1i8, 0, 1];
        let parents = [0i64, 0, 1];
        let mut carry = [0i64; 3];
        let mut parents_out = [0i64; 3];
        let mut outidx = [0i64; 3];
        let n = byte_masked_array_reduce_next_64(
            &mut carry,
            &mut parents_out,
            &mut outidx,
            &mask,
            &parents,
            true,
        );
        assert_eq!(n, 2);
        assert_eq!(&carry[..n], &[0, 2]);
        assert_eq!(&parents_out[..n], &[0, 1]);
        assert_eq!(outidx, [0, -1, 1]);
    }
}
