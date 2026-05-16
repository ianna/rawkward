// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Fill a slice of an index with `base, base+1, base+2, …`
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_fill_count.cpp`.

/// Write `base+i` into `toindex[toindexoffset + i]` for `i` in `0..length`.
///
/// # Examples
///
/// ```
/// use kernels::cpu::index::indexed_fill_count::indexed_array_fill_to64_count;
///
/// let mut to = [0i64; 6];
/// indexed_array_fill_to64_count(&mut to, 2, 3, 10);
/// assert_eq!(to, [0, 0, 10, 11, 12, 0]);
/// ```
pub fn indexed_array_fill_count(
    toindex: &mut [i64],
    toindexoffset: usize,
    length: usize,
    base: i64,
) {
    for i in 0..length {
        toindex[toindexoffset + i] = base + i as i64;
    }
}

/// Typed alias (mirrors `awkward_IndexedArray_fill_to64_count`).
pub fn indexed_array_fill_to64_count(
    toindex: &mut [i64],
    toindexoffset: usize,
    length: usize,
    base: i64,
) {
    indexed_array_fill_count(toindex, toindexoffset, length, base);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut to = [0i64; 6];
        indexed_array_fill_to64_count(&mut to, 2, 3, 10);
        assert_eq!(to, [0, 0, 10, 11, 12, 0]);
    }

    #[test]
    fn offset_zero() {
        let mut to = [0i64; 4];
        indexed_array_fill_to64_count(&mut to, 0, 4, 0);
        assert_eq!(to, [0, 1, 2, 3]);
    }

    #[test]
    fn zero_length() {
        let mut to = [99i64; 3];
        indexed_array_fill_to64_count(&mut to, 1, 0, 5);
        assert_eq!(to, [99, 99, 99]); // unchanged
    }
}
