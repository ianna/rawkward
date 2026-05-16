// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Shrink a jagged slice by filtering out missing entries.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_getitem_jagged_shrink.cpp`.

/// For each outer list `i` in `slicestarts/slicestops`, scan `missing[j]` for
/// `j` in `slicestarts[i]..slicestops[i]`:
/// * Non-negative `missing[j]` → `tocarry[k++] = j`; increment `smallcount`.
/// * Negative → skip.
///
/// Writes:
/// * `tosmalloffsets[i+1] = tosmalloffsets[i] + smallcount`
/// * `tolargeoffsets[i+1] = tolargeoffsets[i] + (slicestop - slicestart)`
///
/// For empty input (`length == 0`), both offsets[0] are set to `0`.  
/// Otherwise `offsets[0] = slicestarts[0]`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_array_getitem_jagged_shrink::list_array_getitem_jagged_shrink_64;
///
/// let slicestarts = [0i64, 3];
/// let slicestops  = [3i64, 5];
/// let missing     = [0i64, -1, 2, -1, 4]; // non-neg at 0,2,4
/// let mut carry        = [0i64; 5];
/// let mut smalloffsets = [0i64; 3];
/// let mut largeoffsets = [0i64; 3];
/// let n = list_array_getitem_jagged_shrink_64(
///     &mut carry, &mut smalloffsets, &mut largeoffsets,
///     &slicestarts, &slicestops, &missing,
/// );
/// assert_eq!(n, 3);
/// assert_eq!(&carry[..n], &[0, 2, 4]);
/// assert_eq!(smalloffsets, [0, 2, 3]);
/// assert_eq!(largeoffsets, [0, 3, 5]);
/// ```
pub fn list_array_getitem_jagged_shrink_64(
    tocarry: &mut [i64],
    tosmalloffsets: &mut [i64],
    tolargeoffsets: &mut [i64],
    slicestarts: &[i64],
    slicestops: &[i64],
    missing: &[i64],
) -> usize {
    let length = slicestarts.len();
    assert_eq!(slicestops.len(), length);
    assert!(tosmalloffsets.len() > length);
    assert!(tolargeoffsets.len() > length);

    let mut k = 0usize;
    if length == 0 {
        tosmalloffsets[0] = 0;
        tolargeoffsets[0] = 0;
        return 0;
    }
    tosmalloffsets[0] = slicestarts[0];
    tolargeoffsets[0] = slicestarts[0];

    for i in 0..length {
        let slicestart = slicestarts[i];
        let slicestop = slicestops[i];
        if slicestart != slicestop {
            let mut smallcount = 0i64;
            for j in slicestart..slicestop {
                if missing[j as usize] >= 0 {
                    tocarry[k] = j;
                    k += 1;
                    smallcount += 1;
                }
            }
            tosmalloffsets[i + 1] = tosmalloffsets[i] + smallcount;
        } else {
            tosmalloffsets[i + 1] = tosmalloffsets[i];
        }
        tolargeoffsets[i + 1] = tolargeoffsets[i] + (slicestop - slicestart);
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let slicestarts = [0i64, 3];
        let slicestops = [3i64, 5];
        let missing = [0i64, -1, 2, -1, 4];
        let mut carry = [0i64; 5];
        let mut small = [0i64; 3];
        let mut large = [0i64; 3];
        let n = list_array_getitem_jagged_shrink_64(
            &mut carry,
            &mut small,
            &mut large,
            &slicestarts,
            &slicestops,
            &missing,
        );
        assert_eq!(n, 3);
        assert_eq!(&carry[..n], &[0, 2, 4]);
        assert_eq!(small, [0, 2, 3]);
        assert_eq!(large, [0, 3, 5]);
    }

    #[test]
    fn all_missing() {
        let slicestarts = [0i64];
        let slicestops = [3i64];
        let missing = [-1i64, -1, -1];
        let mut carry = [0i64; 3];
        let mut small = [0i64; 2];
        let mut large = [0i64; 2];
        let n = list_array_getitem_jagged_shrink_64(
            &mut carry,
            &mut small,
            &mut large,
            &slicestarts,
            &slicestops,
            &missing,
        );
        assert_eq!(n, 0);
        assert_eq!(small, [0, 0]);
        assert_eq!(large, [0, 3]);
    }

    #[test]
    fn empty_input() {
        let mut carry: [i64; 0] = [];
        let mut small = [0i64; 1];
        let mut large = [0i64; 1];
        let n =
            list_array_getitem_jagged_shrink_64(&mut carry, &mut small, &mut large, &[], &[], &[]);
        assert_eq!(n, 0);
        assert_eq!(small[0], 0);
        assert_eq!(large[0], 0);
    }

    #[test]
    fn empty_slice_list() {
        // slicestart == slicestop → skip, counts unchanged
        let slicestarts = [2i64];
        let slicestops = [2i64];
        let missing: [i64; 0] = [];
        let mut carry: [i64; 0] = [];
        let mut small = [0i64; 2];
        let mut large = [0i64; 2];
        let n = list_array_getitem_jagged_shrink_64(
            &mut carry,
            &mut small,
            &mut large,
            &slicestarts,
            &slicestops,
            &missing,
        );
        assert_eq!(n, 0);
        // tosmalloffsets[0]=slicestarts[0]=2; [1]=2+0=2
        assert_eq!(small, [2, 2]);
        assert_eq!(large, [2, 2]);
    }
}
