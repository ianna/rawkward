// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Fill a flat index with within-list positions using an offsets array.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_localindex.cpp`.

/// For each list `i` spanning `offsets[i]..offsets[i+1]`, write `j - offsets[i]`
/// into `toindex[j]`.
pub fn list_array_localindex<C>(toindex: &mut [i64], offsets: &[C])
where
    C: Copy + Into<i64>,
{
    let length = offsets.len().saturating_sub(1);
    for i in 0..length {
        let start: i64 = offsets[i].into();
        let stop: i64 = offsets[i + 1].into();
        for j in start..stop {
            toindex[j as usize] = j - start;
        }
    }
}

pub fn list_array32_localindex_64(toindex: &mut [i64], offsets: &[i32]) {
    list_array_localindex(toindex, offsets);
}
pub fn list_array_u32_localindex_64(toindex: &mut [i64], offsets: &[u32]) {
    list_array_localindex(toindex, offsets);
}
pub fn list_array64_localindex_64(toindex: &mut [i64], offsets: &[i64]) {
    list_array_localindex(toindex, offsets);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_lists() {
        // offsets [0,3,5] → list0=[0,1,2], list1=[3,4]
        let offsets = [0i64, 3, 5];
        let mut out = [0i64; 5];
        list_array64_localindex_64(&mut out, &offsets);
        assert_eq!(out, [0, 1, 2, 0, 1]);
    }

    #[test]
    fn empty_list() {
        let offsets = [0i64, 0, 2];
        let mut out = [0i64; 2];
        list_array64_localindex_64(&mut out, &offsets);
        assert_eq!(out, [0, 1]);
    }

    #[test]
    fn single_element_lists() {
        let offsets = [0i64, 1, 2, 3];
        let mut out = [0i64; 3];
        list_array64_localindex_64(&mut out, &offsets);
        assert_eq!(out, [0, 0, 0]);
    }
}
