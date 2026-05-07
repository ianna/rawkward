// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Two-pass rearrangement of a shifted index array.
//!
//! Corresponds to `src/cpu-kernels/awkward_NumpyArray_rearrange_shifted.cpp`.

/// Apply a two-pass adjustment to `toptr`:
///
/// **Pass 1** (offset by `fromoffsets`): for each list segment `i` in
/// `0..offsetslength-1`, add `fromoffsets[i]` to every element in the
/// corresponding segment of `toptr`.
///
/// **Pass 2** (shift + de-start): for each flat position `i`:
/// `toptr[i] += fromshifts[toptr[i]] - starts[fromparents[i]]`.
///
/// Both `fromshifts` and `fromoffsets` are `i64` arrays; `toptr` is mutated
/// in-place.
<<<<<<< HEAD
#[inline]
=======
>>>>>>> origin/main
pub fn numpy_array_rearrange_shifted(
    toptr: &mut [i64],
    fromshifts: &[i64],
    fromoffsets: &[i64],
    fromparents: &[i64],
    fromstarts: &[i64],
) {
    // Pass 1: add fromoffsets[i] to each element in the i-th segment.
<<<<<<< HEAD
    // Slice-form lets the compiler vectorize the per-segment add.
=======
>>>>>>> origin/main
    let mut k = 0usize;
    let nlists = fromoffsets.len().saturating_sub(1);
    for i in 0..nlists {
        let seg_len = (fromoffsets[i + 1] - fromoffsets[i]) as usize;
<<<<<<< HEAD
        let off = fromoffsets[i];
        for slot in toptr[k..k + seg_len].iter_mut() {
            *slot += off;
        }
        k += seg_len;
    }
    // Pass 2: apply shifts and de-start.
    for (slot, &parent) in toptr.iter_mut().zip(fromparents.iter()) {
        let start = fromstarts[parent as usize];
        let idx = *slot as usize;
        *slot += fromshifts[idx] - start;
=======
        for _ in 0..seg_len {
            toptr[k] += fromoffsets[i];
            k += 1;
        }
    }
    // Pass 2: apply shifts and de-start.
    let length = fromparents.len();
    for i in 0..length {
        let parent = fromparents[i] as usize;
        let start = fromstarts[parent];
        let idx = toptr[i] as usize;
        toptr[i] += fromshifts[idx] - start;
>>>>>>> origin/main
    }
}

/// Typed alias (mirrors `awkward_NumpyArray_rearrange_shifted_toint64_fromint64`).
pub fn numpy_array_rearrange_shifted_toint64_fromint64(
    toptr: &mut [i64],
    fromshifts: &[i64],
    fromoffsets: &[i64],
    fromparents: &[i64],
    fromstarts: &[i64],
) {
    numpy_array_rearrange_shifted(toptr, fromshifts, fromoffsets, fromparents, fromstarts);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_case() {
        // Single list of 3 elements, offset=0, shifts=all-zero, starts=[0].
        let fromoffsets = [0i64, 3];
        let fromshifts = [0i64, 0, 0];
        let fromparents = [0i64, 0, 0];
        let fromstarts = [0i64];
        let mut toptr = [0i64, 1, 2];
        numpy_array_rearrange_shifted_toint64_fromint64(
            &mut toptr,
            &fromshifts,
            &fromoffsets,
            &fromparents,
            &fromstarts,
        );
        // Pass1: toptr += 0 (offset[0]=0) → unchanged.
        // Pass2: toptr[i] += shifts[toptr[i]] - starts[0] = 0 - 0 = 0 → unchanged.
        assert_eq!(toptr, [0, 1, 2]);
    }

    #[test]
    fn offset_applied() {
        // Single list, offset=5, shifts=all-zero, starts=[0].
        // After pass1: toptr = [5+0, 5+1] = [5,6].
        // Pass2: toptr[i] += shifts[toptr[i]] - 0.
        let fromoffsets = [5i64, 7];
        let fromshifts = [0i64; 10];
        let fromparents = [0i64, 0];
        let fromstarts = [0i64];
        let mut toptr = [0i64, 1];
        numpy_array_rearrange_shifted_toint64_fromint64(
            &mut toptr,
            &fromshifts,
            &fromoffsets,
            &fromparents,
            &fromstarts,
        );
        assert_eq!(toptr, [5, 6]);
    }
}
