// Copyright (c) 2026 Ianna Osborne 
// SPDX-License-Identifier: BSD-3-Clause

//! Deduplicate consecutive equal strings in a flat byte array.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_NumpyArray_unique_strings_uint8.cpp`.

/// Scan `toptr` (strings defined by `offsets`) and keep only strings that
/// differ from the previous one.  `toptr` is rewritten in-place.
///
/// # Returns
///
/// The number of unique strings written into `outoffsets`.
pub fn numpy_array_unique_strings_uint8(
    toptr: &mut [u8],
    offsets: &[i64],
    offsetslength: usize,
    outoffsets: &mut [i64],
) -> usize {
    let mut slen: i64 = 0;
    let mut index = 0usize;
    let mut counter = 0usize;
    let mut start = 0usize;
    outoffsets[counter] = offsets[0];
    counter += 1;

    for i in 0..offsetslength.saturating_sub(1) {
        let cur_len = offsets[i + 1] - offsets[i];
        let mut differ = cur_len != slen;
        if !differ {
            let mut k = 0usize;
            for j in offsets[i] as usize..offsets[i + 1] as usize {
                if toptr[start + k] != toptr[j] { differ = true; break; }
                k += 1;
            }
        }
        if differ {
            for j in offsets[i] as usize..offsets[i + 1] as usize {
                toptr[index] = toptr[j];
                index += 1;
            }
            start = offsets[i] as usize;
            outoffsets[counter] = index as i64;
            counter += 1;
        }
        slen = cur_len;
    }
    counter
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_duplicates() {
        let mut data   = b"aabbcc".to_vec();
        let offsets    = [0i64, 2, 4, 6];
        let mut outoff = [0i64; 4];
        let n = numpy_array_unique_strings_uint8(&mut data, &offsets, 4, &mut outoff);
        // "aa","bb","cc" are all distinct → 3 unique
        assert_eq!(n, 4); // outoffsets has 4 entries for 3 strings
    }

    #[test]
    fn keeps_first_of_run() {
        let mut data   = b"aaaa".to_vec();
        let offsets    = [0i64, 2, 4];
        let mut outoff = [0i64; 3];
        let n = numpy_array_unique_strings_uint8(&mut data, &offsets, 3, &mut outoff);
        assert_eq!(n, 2); // only 1 unique string
        assert_eq!(outoff[1], 2);
    }
}
