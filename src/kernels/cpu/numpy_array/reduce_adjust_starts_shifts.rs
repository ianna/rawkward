// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Adjust argmin/argmax results incorporating a shifts array.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_NumpyArray_reduce_adjust_starts_shifts_64.cpp`.

/// Like [`numpy_array_reduce_adjust_starts_64`] but also adds `shifts[i]`:
/// `toptr[k] += shifts[toptr[k]] - starts[parents[toptr[k]]]`
///
/// [`numpy_array_reduce_adjust_starts_64`]: crate::numpy_array_reduce_adjust_starts_64::numpy_array_reduce_adjust_starts_64
pub fn numpy_array_reduce_adjust_starts_shifts_64(
    toptr: &mut [i64],
    parents: &[i64],
    starts: &[i64],
    shifts: &[i64],
) {
    for v in toptr.iter_mut() {
        let i = *v;
        if i >= 0 {
            let parent = parents[i as usize] as usize;
            let start = starts[parent];
            *v += shifts[i as usize] - start;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_shifts() {
        // toptr[0]=2 → parents[2]=0, starts[0]=0, shifts[2]=1 → 2+1-0=3
        let parents = [0i64, 0, 0];
        let starts = [0i64];
        let shifts = [0i64, 0, 1];
        let mut toptr = [2i64];
        numpy_array_reduce_adjust_starts_shifts_64(&mut toptr, &parents, &starts, &shifts);
        assert_eq!(toptr, [3]);
    }

    #[test]
    fn minus_one_unchanged() {
        let parents = [0i64];
        let starts = [0i64];
        let shifts = [0i64];
        let mut toptr = [-1i64];
        numpy_array_reduce_adjust_starts_shifts_64(&mut toptr, &parents, &starts, &shifts);
        assert_eq!(toptr, [-1]);
    }
}
