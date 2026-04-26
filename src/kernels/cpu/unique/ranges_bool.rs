// Copyright (c) 2026 Ianna Osborne 
// SPDX-License-Identifier: BSD-3-Clause

//! Deduplicate consecutive equal booleans within each group, in-place.
//!
//! Corresponds to `src/cpu-kernels/awkward_unique_ranges_bool.cpp`.
//!
//! Uses truthy comparison (`!= 0`) rather than bit equality, matching the C++ behaviour.

/// Like `unique_ranges` but compares booleans as truthy values.
pub fn unique_ranges_bool(
    toptr: &mut [bool],
    fromoffsets: &[i64],
    offsetslength: usize,
    tooffsets: &mut [i64],
) {
    assert!(tooffsets.len() >= offsetslength);
    let mut m = 0usize;
    for i in 0..offsetslength.saturating_sub(1) {
        tooffsets[i] = m as i64;
        let start = fromoffsets[i] as usize;
        let stop  = fromoffsets[i + 1] as usize;
        if start >= stop { continue; }
        toptr[m] = toptr[start];
        m += 1;
        for k in start..stop {
            if toptr[m - 1] != toptr[k] {
                toptr[m] = toptr[k];
                m += 1;
            }
        }
    }
    if offsetslength > 0 {
        tooffsets[offsetslength - 1] = m as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut data    = [true, true, false, true];
        let fromoffsets = [0i64, 4];
        let mut outoff  = [0i64; 2];
        unique_ranges_bool(&mut data, &fromoffsets, 2, &mut outoff);
        // true, false, true → 3 unique (consecutive differ)
        assert_eq!(outoff, [0, 3]);
    }

    #[test]
    fn all_same() {
        let mut data    = [false, false, false];
        let fromoffsets = [0i64, 3];
        let mut outoff  = [0i64; 2];
        unique_ranges_bool(&mut data, &fromoffsets, 2, &mut outoff);
        assert_eq!(outoff, [0, 1]);
    }
}
