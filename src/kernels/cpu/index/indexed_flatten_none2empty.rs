// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Flatten an IndexedOptionArray by replacing None with empty lists.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_flatten_none2empty.cpp`.

use crate::kernels::cpu::error::KernelError;

/// Build `outoffsets` for flattening: None entries get zero-length lists,
/// valid entries get the list lengths from `offsets[idx]..offsets[idx+1]`.
///
/// `outoffsets` must have length `outindex.len() + 1`.
///
/// # Errors
///
/// Returns an error if `idx + 1 >= offsetslength` for a non-null entry.
pub fn indexed_array_flatten_none2empty<C>(
    outoffsets: &mut [i64],
    outindex: &[C],
    offsets: &[i64],
) -> Result<(), KernelError>
where
    C: Copy + Into<i64>,
{
    let outindexlength = outindex.len();
    let offsetslength = offsets.len() as i64;
    assert!(outoffsets.len() > outindexlength);
    outoffsets[0] = offsets[0];
    for i in 0..outindexlength {
        let idx: i64 = outindex[i].into();
        if idx < 0 {
            outoffsets[i + 1] = outoffsets[i];
        } else if idx + 1 >= offsetslength {
            return Err(KernelError::at("flattening offset out of range", i as i64));
        } else {
            let count = offsets[(idx + 1) as usize] - offsets[idx as usize];
            outoffsets[i + 1] = outoffsets[i] + count;
        }
    }
    Ok(())
}

pub fn indexed_array32_flatten_none2empty_64(
    outoffsets: &mut [i64],
    outindex: &[i32],
    offsets: &[i64],
) -> Result<(), KernelError> {
    indexed_array_flatten_none2empty(outoffsets, outindex, offsets)
}
pub fn indexed_array_u32_flatten_none2empty_64(
    outoffsets: &mut [i64],
    outindex: &[u32],
    offsets: &[i64],
) -> Result<(), KernelError> {
    indexed_array_flatten_none2empty(outoffsets, outindex, offsets)
}
pub fn indexed_array64_flatten_none2empty_64(
    outoffsets: &mut [i64],
    outindex: &[i64],
    offsets: &[i64],
) -> Result<(), KernelError> {
    indexed_array_flatten_none2empty(outoffsets, outindex, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        // outindex: [0, -1, 1] → lists from offsets[0..1]=2, empty, offsets[1..2]=3
        let outindex = [0i64, -1, 1];
        let offsets = [0i64, 2, 5];
        let mut out = [0i64; 4];
        indexed_array64_flatten_none2empty_64(&mut out, &outindex, &offsets).unwrap();
        assert_eq!(out, [0, 2, 2, 5]);
    }

    #[test]
    fn all_none() {
        let outindex = [-1i64, -1];
        let offsets = [0i64, 3];
        let mut out = [0i64; 3];
        indexed_array64_flatten_none2empty_64(&mut out, &outindex, &offsets).unwrap();
        assert_eq!(out, [0, 0, 0]);
    }

    #[test]
    fn out_of_range_error() {
        let outindex = [5i64]; // 5+1=6 >= offsets.len()=3
        let offsets = [0i64, 2, 4];
        let mut out = [0i64; 2];
        assert!(indexed_array64_flatten_none2empty_64(&mut out, &outindex, &offsets).is_err());
    }
}
