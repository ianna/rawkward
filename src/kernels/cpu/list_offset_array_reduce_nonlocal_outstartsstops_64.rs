// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Build output starts/stops from a distincts array for a non-local reduction.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64.cpp`.

/// Populate `outstarts` and `outstops` from the `distincts` array.
///
/// `distincts` has length `lendistincts = outlength * maxcount` and is laid
/// out as `outlength` consecutive blocks of `maxcount` entries each.  For
/// each block `k`:
/// * `outstarts[k] = start of block k`
/// * `outstops[k]` starts equal to `outstarts[k]` and advances past each
///   non-`-1` entry.
///
/// When `outlength == 0` or `lendistincts == 0`, all starts and stops are 0.
pub fn list_offset_array_reduce_nonlocal_outstartsstops_64(
    outstarts: &mut [i64],
    outstops: &mut [i64],
    distincts: &[i64],
    lendistincts: usize,
    outlength: usize,
) {
    assert!(outstarts.len() >= outlength);
    assert!(outstops.len() >= outlength);

    if outlength == 0 || lendistincts == 0 {
        for k in 0..outlength {
            outstarts[k] = 0;
            outstops[k] = 0;
        }
        return;
    }

    let maxcount = lendistincts / outlength;
    let mut k = 0usize;
    let mut i_next = 0usize;

    for i in 0..lendistincts {
        if i == i_next {
            i_next += maxcount;
            outstarts[k] = i as i64;
            outstops[k] = i as i64;
            k += 1;
        }
        if distincts[i] != -1 {
            outstops[k - 1] = (i + 1) as i64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        // outlength=2, maxcount=3
        // distincts=[0,1,-1, 2,-1,-1]
        // block0: start=0, non-(-1) at 0,1 → stop=2
        // block1: start=3, non-(-1) at 3   → stop=4
        let distincts = [0i64, 1, -1, 2, -1, -1];
        let mut starts = [0i64; 2];
        let mut stops = [0i64; 2];
        list_offset_array_reduce_nonlocal_outstartsstops_64(
            &mut starts,
            &mut stops,
            &distincts,
            6,
            2,
        );
        assert_eq!(starts, [0, 3]);
        assert_eq!(stops, [2, 4]);
    }

    #[test]
    fn all_minus_one() {
        let distincts = [-1i64, -1, -1, -1];
        let mut starts = [0i64; 2];
        let mut stops = [0i64; 2];
        list_offset_array_reduce_nonlocal_outstartsstops_64(
            &mut starts,
            &mut stops,
            &distincts,
            4,
            2,
        );
        assert_eq!(starts, [0, 2]);
        assert_eq!(stops, [0, 2]);
    }

    #[test]
    fn zero_outlength() {
        let distincts: [i64; 0] = [];
        let mut starts: [i64; 0] = [];
        let mut stops: [i64; 0] = [];
        list_offset_array_reduce_nonlocal_outstartsstops_64(
            &mut starts,
            &mut stops,
            &distincts,
            0,
            0,
        );
        // No-op, no panic.
    }

    #[test]
    fn all_present() {
        let distincts = [0i64, 1, 2, 3];
        let mut starts = [0i64; 2];
        let mut stops = [0i64; 2];
        list_offset_array_reduce_nonlocal_outstartsstops_64(
            &mut starts,
            &mut stops,
            &distincts,
            4,
            2,
        );
        assert_eq!(starts, [0, 2]);
        assert_eq!(stops, [2, 4]);
    }
}
