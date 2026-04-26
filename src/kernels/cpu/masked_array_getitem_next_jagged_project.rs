// Copyright (c) 2026 Ianna Osborne 
// SPDX-License-Identifier: BSD-3-Clause

//! Project the non-null entries of a MaskedArray through a jagged getitem.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_MaskedArray_getitem_next_jagged_project.cpp`.

/// For each `i` in `0..length`, if `index[i] >= 0`, copy
/// `starts_in[i]` and `stops_in[i]` into `starts_out[k]` / `stops_out[k]`
/// and increment `k`.  Returns the number of valid entries written.
pub fn masked_array_getitem_next_jagged_project<C>(
    index: &[C],
    starts_in: &[i64],
    stops_in: &[i64],
    starts_out: &mut [i64],
    stops_out: &mut [i64],
) -> usize
where
    C: Copy + Into<i64>,
{
    let length = index.len();
    assert_eq!(starts_in.len(), length);
    assert_eq!(stops_in.len(), length);
    let mut k = 0usize;
    for i in 0..length {
        if index[i].into() >= 0 {
            starts_out[k] = starts_in[i];
            stops_out [k] = stops_in[i];
            k += 1;
        }
    }
    k
}

pub fn masked_array32_getitem_next_jagged_project(index: &[i32], starts_in: &[i64], stops_in: &[i64], starts_out: &mut [i64], stops_out: &mut [i64]) -> usize {
    masked_array_getitem_next_jagged_project(index, starts_in, stops_in, starts_out, stops_out)
}
pub fn masked_array_u32_getitem_next_jagged_project(index: &[u32], starts_in: &[i64], stops_in: &[i64], starts_out: &mut [i64], stops_out: &mut [i64]) -> usize {
    masked_array_getitem_next_jagged_project(index, starts_in, stops_in, starts_out, stops_out)
}
pub fn masked_array64_getitem_next_jagged_project(index: &[i64], starts_in: &[i64], stops_in: &[i64], starts_out: &mut [i64], stops_out: &mut [i64]) -> usize {
    masked_array_getitem_next_jagged_project(index, starts_in, stops_in, starts_out, stops_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let index     = [0i64, -1, 2, -1];
        let starts_in = [0i64, 10, 20, 30];
        let stops_in  = [5i64, 15, 25, 35];
        let mut s_out = [0i64; 4];
        let mut p_out = [0i64; 4];
        let n = masked_array64_getitem_next_jagged_project(&index, &starts_in, &stops_in, &mut s_out, &mut p_out);
        assert_eq!(n, 2);
        assert_eq!(&s_out[..n], &[0, 20]);
        assert_eq!(&p_out[..n], &[5, 25]);
    }

    #[test]
    fn all_null() {
        let index     = [-1i64, -1];
        let starts_in = [0i64, 5];
        let stops_in  = [5i64, 10];
        let mut s = [0i64; 2];
        let mut p = [0i64; 2];
        assert_eq!(masked_array64_getitem_next_jagged_project(&index, &starts_in, &stops_in, &mut s, &mut p), 0);
    }
}
