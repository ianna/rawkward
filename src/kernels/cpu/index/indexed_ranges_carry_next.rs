// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Collect non-null index values from ranges into a carry array.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_ranges_carry_next_64.cpp`.

/// For each range `i` defined by `fromstarts[i]..fromstops[i]`, scan
/// `index[fromstarts[i]..fromstops[i]]` and append every non-negative value
/// to `tocarry`.
///
/// Null entries (`index[j] < 0`) are silently skipped.
///
/// # Returns
///
/// The number of entries written to `tocarry`.
///
/// # Panics
///
/// Panics if `fromstarts.len() != fromstops.len()` or if `tocarry` is too
/// short to hold all valid entries.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_ranges_carry_next_64::indexed_array64_ranges_carry_next_64;
///
/// // Two ranges: index[0..2] = [5, -1]  and  index[2..4] = [-1, 7]
/// let index      = [5i64, -1, -1, 7];
/// let fromstarts = [0i64, 2];
/// let fromstops  = [2i64, 4];
/// let mut carry  = [0i64; 4];
/// let n = indexed_array64_ranges_carry_next_64(&mut carry, &index, &fromstarts, &fromstops);
/// assert_eq!(n, 2);
/// assert_eq!(&carry[..n], &[5, 7]);
/// ```
pub fn indexed_array_ranges_carry_next_64<C>(
    tocarry: &mut [i64],
    index: &[C],
    fromstarts: &[i64],
    fromstops: &[i64],
) -> usize
where
    C: Copy + Into<i64>,
{
    assert_eq!(fromstarts.len(), fromstops.len());
    let mut k = 0usize;
    for i in 0..fromstarts.len() {
        let start = fromstarts[i] as usize;
        let stop = fromstops[i] as usize;
        for j in start..stop {
            let v: i64 = index[j].into();
            if v >= 0 {
                tocarry[k] = v;
                k += 1;
            }
        }
    }
    k
}

pub fn indexed_array32_ranges_carry_next_64(
    tocarry: &mut [i64],
    index: &[i32],
    fromstarts: &[i64],
    fromstops: &[i64],
) -> usize {
    indexed_array_ranges_carry_next_64(tocarry, index, fromstarts, fromstops)
}
pub fn indexed_array_u32_ranges_carry_next_64(
    tocarry: &mut [i64],
    index: &[u32],
    fromstarts: &[i64],
    fromstops: &[i64],
) -> usize {
    indexed_array_ranges_carry_next_64(tocarry, index, fromstarts, fromstops)
}
pub fn indexed_array64_ranges_carry_next_64(
    tocarry: &mut [i64],
    index: &[i64],
    fromstarts: &[i64],
    fromstops: &[i64],
) -> usize {
    indexed_array_ranges_carry_next_64(tocarry, index, fromstarts, fromstops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_mixed() {
        let index = [5i64, -1, -1, 7];
        let fromstarts = [0i64, 2];
        let fromstops = [2i64, 4];
        let mut carry = [0i64; 4];
        let n = indexed_array64_ranges_carry_next_64(&mut carry, &index, &fromstarts, &fromstops);
        assert_eq!(n, 2);
        assert_eq!(&carry[..n], &[5, 7]);
    }

    #[test]
    fn all_null() {
        let index = [-1i64, -1, -1];
        let fromstarts = [0i64];
        let fromstops = [3i64];
        let mut carry = [0i64; 3];
        let n = indexed_array64_ranges_carry_next_64(&mut carry, &index, &fromstarts, &fromstops);
        assert_eq!(n, 0);
    }

    #[test]
    fn all_valid() {
        let index = [10i64, 20, 30];
        let fromstarts = [0i64];
        let fromstops = [3i64];
        let mut carry = [0i64; 3];
        let n = indexed_array64_ranges_carry_next_64(&mut carry, &index, &fromstarts, &fromstops);
        assert_eq!(n, 3);
        assert_eq!(&carry[..n], &[10, 20, 30]);
    }

    #[test]
    fn empty_ranges() {
        let index: [i64; 0] = [];
        let fromstarts = [0i64, 0];
        let fromstops = [0i64, 0];
        let mut carry = [0i64; 0];
        let n = indexed_array64_ranges_carry_next_64(&mut carry, &index, &fromstarts, &fromstops);
        assert_eq!(n, 0);
    }

    #[test]
    fn i32_index() {
        let index = [3i32, -1, 7];
        let fromstarts = [0i64];
        let fromstops = [3i64];
        let mut carry = [0i64; 3];
        let n = indexed_array32_ranges_carry_next_64(&mut carry, &index, &fromstarts, &fromstops);
        assert_eq!(n, 2);
        assert_eq!(&carry[..n], &[3, 7]);
    }
}
