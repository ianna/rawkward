// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Collect all (non-null, in-range) index values as a carry array.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_getitem_nextcarry.cpp`.

use crate::kernels::cpu::error::KernelError;

/// Copy every entry of `fromindex` into `tocarry`. Returns an error if any
/// value is negative or >= `lencontent`.
///
/// # Returns `usize` – number of entries written.
pub fn indexed_array_getitem_nextcarry<C>(
    tocarry: &mut [i64],
    fromindex: &[C],
    lencontent: i64,
) -> Result<usize, KernelError>
where
    C: Copy + Into<i64>,
{
    for (i, &v) in fromindex.iter().enumerate() {
        let j: i64 = v.into();
        if j < 0 || j >= lencontent {
            return Err(KernelError::new("index out of range", i as i64, j));
        }
        tocarry[i] = j;
    }
    Ok(fromindex.len())
}

pub fn indexed_array32_getitem_nextcarry_64(
    tocarry: &mut [i64],
    fromindex: &[i32],
    lencontent: i64,
) -> Result<usize, KernelError> {
    indexed_array_getitem_nextcarry(tocarry, fromindex, lencontent)
}
pub fn indexed_array_u32_getitem_nextcarry_64(
    tocarry: &mut [i64],
    fromindex: &[u32],
    lencontent: i64,
) -> Result<usize, KernelError> {
    indexed_array_getitem_nextcarry(tocarry, fromindex, lencontent)
}
pub fn indexed_array64_getitem_nextcarry_64(
    tocarry: &mut [i64],
    fromindex: &[i64],
    lencontent: i64,
) -> Result<usize, KernelError> {
    indexed_array_getitem_nextcarry(tocarry, fromindex, lencontent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid() {
        let from = [0i64, 2, 4];
        let mut carry = [0i64; 3];
        let n = indexed_array64_getitem_nextcarry_64(&mut carry, &from, 5).unwrap();
        assert_eq!(n, 3);
        assert_eq!(carry, [0, 2, 4]);
    }

    #[test]
    fn negative_index_error() {
        let from = [-1i64];
        let mut carry = [0i64; 1];
        assert!(indexed_array64_getitem_nextcarry_64(&mut carry, &from, 5).is_err());
    }

    #[test]
    fn out_of_range_error() {
        let from = [10i64];
        let mut carry = [0i64; 1];
        assert!(indexed_array64_getitem_nextcarry_64(&mut carry, &from, 5).is_err());
    }
}
