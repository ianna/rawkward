// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build the carry index for the local-reduction preparation step.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_IndexedArray_local_preparenext_64.cpp`.

/// For each `i` in `0..parentslength`:
/// * If `j < nextlen` and `parents[i] == nextparents[j]`: `tocarry[i] = j`, `j++`.
/// * Else: `tocarry[i] = -1`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_local_preparenext_64::indexed_array_local_preparenext_64;
///
/// let parents     = [0i64, 0, 1, 2];
/// let nextparents = [0i64, 1];
/// let mut carry   = [0i64; 4];
/// indexed_array_local_preparenext_64(&mut carry, &parents, &nextparents);
/// assert_eq!(carry, [0, -1, 1, -1]);
/// ```
pub fn indexed_array_local_preparenext_64(
    tocarry: &mut [i64],
    parents: &[i64],
    nextparents: &[i64],
) {
    let nextlen = nextparents.len();
    let mut j = 0usize;
    for (i, &p) in parents.iter().enumerate() {
        if j < nextlen && p == nextparents[j] {
            tocarry[i] = j as i64;
            j += 1;
        } else {
            tocarry[i] = -1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let parents = [0i64, 0, 1, 2];
        let nextparents = [0i64, 1];
        let mut carry = [0i64; 4];
        indexed_array_local_preparenext_64(&mut carry, &parents, &nextparents);
        assert_eq!(carry, [0, -1, 1, -1]);
    }

    #[test]
    fn all_match() {
        let parents = [0i64, 1, 2];
        let nextparents = [0i64, 1, 2];
        let mut carry = [0i64; 3];
        indexed_array_local_preparenext_64(&mut carry, &parents, &nextparents);
        assert_eq!(carry, [0, 1, 2]);
    }

    #[test]
    fn no_match() {
        let parents = [0i64, 0];
        let nextparents = [1i64, 2];
        let mut carry = [0i64; 2];
        indexed_array_local_preparenext_64(&mut carry, &parents, &nextparents);
        assert_eq!(carry, [-1, -1]);
    }
}
