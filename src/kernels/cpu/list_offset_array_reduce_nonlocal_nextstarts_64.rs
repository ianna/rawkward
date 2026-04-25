// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Record the starting position of each new parent group in the next-level array.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_nextstarts_64.cpp`.

/// For each position `i` in `nextparents`, if `nextparents[i]` differs from
/// the previous value, write `nextstarts[nextparents[i]] = i`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_offset_array_reduce_nonlocal_nextstarts_64::list_offset_array_reduce_nonlocal_nextstarts_64;
///
/// let nextparents = [0i64, 0, 1, 1, 2];
/// let mut nextstarts = [0i64; 3];
/// list_offset_array_reduce_nonlocal_nextstarts_64(&mut nextstarts, &nextparents);
/// assert_eq!(nextstarts, [0, 2, 4]);
/// ```
pub fn list_offset_array_reduce_nonlocal_nextstarts_64(
    nextstarts: &mut [i64],
    nextparents: &[i64],
) {
    let mut lastnextparent = -1i64;
    for (i, &p) in nextparents.iter().enumerate() {
        if p != lastnextparent {
            nextstarts[p as usize] = i as i64;
        }
        lastnextparent = p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let nextparents = [0i64, 0, 1, 1, 2];
        let mut nextstarts = [0i64; 3];
        list_offset_array_reduce_nonlocal_nextstarts_64(&mut nextstarts, &nextparents);
        assert_eq!(nextstarts, [0, 2, 4]);
    }

    #[test]
    fn each_element_different_parent() {
        let nextparents = [0i64, 1, 2];
        let mut nextstarts = [0i64; 3];
        list_offset_array_reduce_nonlocal_nextstarts_64(&mut nextstarts, &nextparents);
        assert_eq!(nextstarts, [0, 1, 2]);
    }

    #[test]
    fn all_same_parent() {
        let nextparents = [0i64, 0, 0, 0];
        let mut nextstarts = [0i64; 1];
        list_offset_array_reduce_nonlocal_nextstarts_64(&mut nextstarts, &nextparents);
        assert_eq!(nextstarts[0], 0);
    }
}
