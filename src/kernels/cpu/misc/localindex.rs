// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Fill an array with consecutive integers starting at zero.
//!
//! Corresponds to `src/cpu-kernels/awkward_localindex.cpp`.

/// Fill `toindex` with the values `[0, 1, 2, …, toindex.len()-1]`.
///
/// Equivalent to `std::iota` from C++, producing the local (within-list)
/// position index used by `LocalIndex` nodes.
///
/// # Type parameters
///
/// `T` must be constructable from `usize` (via `TryFrom`).
///
/// # Panics
///
/// Panics if any index value overflows `T` (only possible for very large
/// arrays with narrow output types).
///
/// # Examples
///
/// ```
/// use kernels::cpu::misc::localindex;
///
/// let mut idx = [0i64; 5];
/// localindex_64(&mut idx);
/// assert_eq!(idx, [0, 1, 2, 3, 4]);
/// ```
pub fn localindex<T>(toindex: &mut [T])
where
    T: TryFrom<usize> + Copy,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    for (i, slot) in toindex.iter_mut().enumerate() {
        *slot = T::try_from(i).expect("index fits in T");
    }
}

/// Typed wrapper for `i64` output
/// (mirrors `awkward_localindex_64`).
pub fn localindex_64(toindex: &mut [i64]) {
    localindex(toindex);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_i64() {
        let mut idx = [0i64; 5];
        localindex_64(&mut idx);
        assert_eq!(idx, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn single_element() {
        let mut idx = [99i64; 1];
        localindex_64(&mut idx);
        assert_eq!(idx[0], 0);
    }

    #[test]
    fn empty_slice() {
        let mut idx: [i64; 0] = [];
        localindex_64(&mut idx); // must not panic
    }

    #[test]
    fn i32_output() {
        let mut idx = [0i32; 4];
        localindex(&mut idx);
        assert_eq!(idx, [0, 1, 2, 3]);
    }

    #[test]
    fn large_length() {
        let mut idx = vec![0i64; 1000];
        localindex_64(&mut idx);
        for (i, &v) in idx.iter().enumerate() {
            assert_eq!(v, i as i64);
        }
    }
}
