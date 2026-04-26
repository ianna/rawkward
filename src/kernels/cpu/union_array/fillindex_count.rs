// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Fill a slice of an index with `0, 1, 2, …` (sequential IDs).
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_fillindex_count.cpp`.

/// Write `toindex[toindexoffset + i] = i` for `i` in `0..length`.
pub fn union_array_fillindex_count(toindex: &mut [i64], toindexoffset: usize, length: usize) {
    for i in 0..length {
        toindex[toindexoffset + i] = i as i64;
    }
}

/// Alias matching `awkward_UnionArray_fillindex_to64_count`.
pub fn union_array_fillindex_to64_count(toindex: &mut [i64], toindexoffset: usize, length: usize) {
    union_array_fillindex_count(toindex, toindexoffset, length);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut to = [0i64; 6];
        union_array_fillindex_to64_count(&mut to, 2, 3);
        assert_eq!(to, [0, 0, 0, 1, 2, 0]);
    }

    #[test]
    fn zero_length() {
        let mut to = [99i64; 3];
        union_array_fillindex_to64_count(&mut to, 1, 0);
        assert_eq!(to, [99, 99, 99]);
    }
}
