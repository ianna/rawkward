// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Assign parent group indices to every flat content element.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_reduce_local_nextparents_64.cpp`.

/// For each list `i` (0-based), fill every position `j` in
/// `nextparents[offsets[i]-offsets[0] .. offsets[i+1]-offsets[0]]` with `i`.
///
/// The `offsets[0]` subtraction handles non-zero-based offsets arrays.
/// Writing stops when `j >= nextparents_length`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_offset_array_reduce_local_nextparents_64::list_offset_array64_reduce_local_nextparents_64;
///
/// let offsets = [0i64, 3, 5, 8];
/// let mut nextparents = [0i64; 8];
/// list_offset_array64_reduce_local_nextparents_64(&mut nextparents, &offsets, 3, 8);
/// assert_eq!(nextparents, [0, 0, 0, 1, 1, 2, 2, 2]);
/// ```
pub fn list_offset_array_reduce_local_nextparents_64<C>(
    nextparents: &mut [i64],
    offsets: &[C],
    length: usize,
    nextparents_length: usize,
) where
    C: Copy + Into<i64>,
{
    assert!(offsets.len() >= length + 1);
    let initial_offset: i64 = offsets[0].into();
    for i in 0..length {
        let start = offsets[i].into() - initial_offset;
        let stop = offsets[i + 1].into() - initial_offset;
        let mut j = start;
        while j < stop && (j as usize) < nextparents_length {
            nextparents[j as usize] = i as i64;
            j += 1;
        }
    }
}

pub fn list_offset_array32_reduce_local_nextparents_64(
    nextparents: &mut [i64],
    offsets: &[i32],
    length: usize,
    nextparents_length: usize,
) {
    list_offset_array_reduce_local_nextparents_64(nextparents, offsets, length, nextparents_length);
}
pub fn list_offset_array_u32_reduce_local_nextparents_64(
    nextparents: &mut [i64],
    offsets: &[u32],
    length: usize,
    nextparents_length: usize,
) {
    list_offset_array_reduce_local_nextparents_64(nextparents, offsets, length, nextparents_length);
}
pub fn list_offset_array64_reduce_local_nextparents_64(
    nextparents: &mut [i64],
    offsets: &[i64],
    length: usize,
    nextparents_length: usize,
) {
    list_offset_array_reduce_local_nextparents_64(nextparents, offsets, length, nextparents_length);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let offsets = [0i64, 3, 5, 8];
        let mut np = [0i64; 8];
        list_offset_array64_reduce_local_nextparents_64(&mut np, &offsets, 3, 8);
        assert_eq!(np, [0, 0, 0, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn non_zero_initial_offset() {
        // offsets start at 2 — positions are relative
        let offsets = [2i64, 4, 7];
        let mut np = [0i64; 5];
        list_offset_array64_reduce_local_nextparents_64(&mut np, &offsets, 2, 5);
        assert_eq!(np, [0, 0, 1, 1, 1]);
    }

    #[test]
    fn clamp_to_nextparents_length() {
        let offsets = [0i64, 10];
        let mut np = [0i64; 3]; // shorter than list length
        list_offset_array64_reduce_local_nextparents_64(&mut np, &offsets, 1, 3);
        assert_eq!(np, [0, 0, 0]);
    }

    #[test]
    fn i32_offsets() {
        let offsets = [0i32, 2, 4];
        let mut np = [0i64; 4];
        list_offset_array32_reduce_local_nextparents_64(&mut np, &offsets, 2, 4);
        assert_eq!(np, [0, 0, 1, 1]);
    }
}
