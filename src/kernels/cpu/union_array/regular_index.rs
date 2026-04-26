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
pub fn union_array_regular_index_64(toindex: &mut [i64], size: usize, fromtags: &[i64]) {
    let mut current = vec![0i64; size];
    for (i, &tag) in fromtags.iter().enumerate() {
        let t = tag as usize;
        toindex[i] = current[t];
        current[t] += 1;
    }
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
pub fn union_array8_64_regular_index(toindex: &mut [i64], size: usize, fromtags: &[i8]) {
    let mut current = vec![0i64; size];
    for (i, &tag) in fromtags.iter().enumerate() {
        let t = tag as usize;
        toindex[i] = current[t];
        current[t] += 1;
    }
}

/// Build a regular i32 index from i8 tags.
pub fn union_array8_32_regular_index(toindex: &mut [i32], size: usize, fromtags: &[i8]) {
    let mut current = vec![0i32; size];
    for (i, &tag) in fromtags.iter().enumerate() {
        let t = tag as usize;
        toindex[i] = current[t];
        current[t] += 1;
    }
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
}