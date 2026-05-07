// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Assign parent row indices to every flat element for a local reduction.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_RegularArray_reduce_local_nextparents_64.cpp`.

/// For each row `i` and slot `j` in `0..size`:
/// `nextparents[i*size + j] = i`
///
/// Every element in a row shares its row index as its parent.  This is the
/// RegularArray analogue of
/// [`list_offset_array_reduce_local_nextparents_64`].
///
/// # Examples
///
/// ```
/// use cpu_kernels::regular_array_reduce_local_nextparents_64::regular_array_reduce_local_nextparents_64;
///
/// let mut np = [0i64; 6];
/// regular_array_reduce_local_nextparents_64(&mut np, 3, 2);
/// assert_eq!(np, [0, 0, 0, 1, 1, 1]);
/// ```
pub fn regular_array_reduce_local_nextparents_64(
    nextparents: &mut [i64],
    size: usize,
    length: usize,
) {
    assert!(nextparents.len() >= length * size);
    let mut k = 0usize;
    for i in 0..length {
        for _j in 0..size {
            nextparents[k] = i as i64;
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut np = [0i64; 6];
        regular_array_reduce_local_nextparents_64(&mut np, 3, 2);
        assert_eq!(np, [0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn size_one() {
        let mut np = [0i64; 4];
        regular_array_reduce_local_nextparents_64(&mut np, 1, 4);
        assert_eq!(np, [0, 1, 2, 3]);
    }

    #[test]
    fn single_row() {
        let mut np = [0i64; 5];
        regular_array_reduce_local_nextparents_64(&mut np, 5, 1);
        assert_eq!(np, [0, 0, 0, 0, 0]);
    }

    #[test]
    fn zero_length() {
        let mut np: [i64; 0] = [];
        regular_array_reduce_local_nextparents_64(&mut np, 3, 0);
    }
}
