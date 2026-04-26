// Copyright (c) 2026 Ianna Osborne 
// SPDX-License-Identifier: BSD-3-Clause

//! Sort a flat byte array interpreted as a sequence of strings.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_NumpyArray_sort_asstrings_uint8.cpp`.

/// Sort strings (defined by `offsets`) in-place, writing sorted bytes to
/// `toptr` and updated offsets to `outoffsets`.
pub fn numpy_array_sort_asstrings_uint8(
    toptr: &mut [u8],
    fromptr: &[u8],
    offsets: &[i64],
    offsetslength: usize,
    outoffsets: &mut [i64],
    ascending: bool,
    stable: bool,
) {
    let nstrings = offsetslength.saturating_sub(1);
    // Collect string slices.
    let mut words: Vec<Vec<u8>> = (0..nstrings).map(|k| {
        let s = offsets[k] as usize;
        let e = offsets[k + 1] as usize;
        fromptr[s..e].to_vec()
    }).collect();

    if stable {
        if ascending { words.sort_by(|a, b| a.cmp(b)); }
        else         { words.sort_by(|a, b| b.cmp(a)); }
    } else if ascending {
        words.sort_unstable_by(|a, b| a.cmp(b));
    } else {
        words.sort_unstable_by(|a, b| b.cmp(a));
    }

    let mut k = 0usize;
    for w in &words {
        for &c in w { toptr[k] = c; k += 1; }
    }
    outoffsets[0] = 0;
    for (o, w) in words.iter().enumerate() {
        outoffsets[o + 1] = outoffsets[o] + w.len() as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascending() {
        let from    = b"bananaapple";   // "banana","apple"
        let offsets = [0i64, 6, 11];
        let mut to      = vec![0u8; 11];
        let mut outoff  = [0i64; 3];
        numpy_array_sort_asstrings_uint8(&mut to, from, &offsets, 3, &mut outoff, true, true);
        let s1 = &to[..outoff[1] as usize];
        let s2 = &to[outoff[1] as usize..outoff[2] as usize];
        assert_eq!(s1, b"apple");
        assert_eq!(s2, b"banana");
    }
}
