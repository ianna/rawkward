// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build a regular per-tag index for a UnionArray.
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_regular_index.cpp`.
//!
//! For each element `i`, assigns `toindex[i] = current[fromtags[i]]` then
//! increments `current[fromtags[i]]`.  This produces a sequential per-content
//! counter, making it easy to reconstruct which position within each variant's
//! content each union element refers to.

/// UnionArrays in awkward never have many variants in practice (usually
/// ≤ 4, never anywhere near this). Keeping `current` on the stack up to
/// this size avoids one heap allocation per call, which matters when these
/// kernels are invoked once per record-batch.
const STACK_SIZE_LIMIT: usize = 32;

use core::ops::AddAssign;

/// Generic core. `T` is the output index type (`i32`/`i64`); `Tag` is the
/// input tag type (`i8`/`i64`/...).
#[inline]
fn regular_index<T, Tag>(toindex: &mut [T], size: usize, fromtags: &[Tag])
where
    T: Copy + Default + AddAssign + From<u8>,
    Tag: Copy + Into<i64>,
{
    let one: T = T::from(1u8);
    if size <= STACK_SIZE_LIMIT {
        let mut current = [T::default(); STACK_SIZE_LIMIT];
        for (slot, &tag) in toindex.iter_mut().zip(fromtags.iter()) {
            let t = Into::<i64>::into(tag) as usize;
            *slot = current[t];
            current[t] += one;
        }
    } else {
        // Cold fallback: extremely unusual, but support it for completeness.
        let mut current: Vec<T> = vec![T::default(); size];
        for (slot, &tag) in toindex.iter_mut().zip(fromtags.iter()) {
            let t = Into::<i64>::into(tag) as usize;
            *slot = current[t];
            current[t] += one;
        }
    }
}

/// Build a regular i64 index from i64 tags.
///
/// # Examples
///
/// ```
/// use cpu_kernels::union_array_regular_index::union_array_regular_index_64;
///
/// let tags = [0i64, 1, 0, 1, 0];
/// let mut idx = [0i64; 5];
/// union_array_regular_index_64(&mut idx, 2, &tags);
/// assert_eq!(idx, [0, 0, 1, 1, 2]);
/// ```
#[inline]
pub fn union_array_regular_index_64(toindex: &mut [i64], size: usize, fromtags: &[i64]) {
    regular_index(toindex, size, fromtags);
}

/// Build a regular i64 index from i8 tags.
///
/// # Examples
///
/// ```
/// use cpu_kernels::union_array_regular_index::union_array8_64_regular_index;
///
/// let tags = [0i8, 0, 1, 0];
/// let mut idx = [0i64; 4];
/// union_array8_64_regular_index(&mut idx, 2, &tags);
/// assert_eq!(idx, [0, 1, 0, 2]);
/// ```
#[inline]
pub fn union_array8_64_regular_index(toindex: &mut [i64], size: usize, fromtags: &[i8]) {
    regular_index(toindex, size, fromtags);
}

/// Build a regular i32 index from i8 tags.
#[inline]
pub fn union_array8_32_regular_index(toindex: &mut [i32], size: usize, fromtags: &[i8]) {
    regular_index(toindex, size, fromtags);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_contents_i64_tags() {
        let tags = [0i64, 1, 0, 1, 0];
        let mut idx = [0i64; 5];
        union_array_regular_index_64(&mut idx, 2, &tags);
        assert_eq!(idx, [0, 0, 1, 1, 2]);
    }

    #[test]
    fn i8_tags() {
        let tags = [0i8, 0, 1, 0];
        let mut idx = [0i64; 4];
        union_array8_64_regular_index(&mut idx, 2, &tags);
        assert_eq!(idx, [0, 1, 0, 2]);
    }

    #[test]
    fn single_content() {
        let tags = [0i64; 4];
        let mut idx = [0i64; 4];
        union_array_regular_index_64(&mut idx, 1, &tags);
        assert_eq!(idx, [0, 1, 2, 3]);
    }

    #[test]
    fn i32_output() {
        let tags = [0i8, 1, 0];
        let mut idx = [0i32; 3];
        union_array8_32_regular_index(&mut idx, 2, &tags);
        assert_eq!(idx, [0, 0, 1]);
    }

    #[test]
    fn fallback_when_size_exceeds_stack_limit() {
        // Take the heap path. Tags only reference content 0, but `size` is large.
        let tags = [0i64, 0, 0];
        let mut idx = [0i64; 3];
        union_array_regular_index_64(&mut idx, STACK_SIZE_LIMIT + 1, &tags);
        assert_eq!(idx, [0, 1, 2]);
    }
}
