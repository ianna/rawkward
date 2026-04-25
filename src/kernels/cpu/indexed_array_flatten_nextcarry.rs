// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Collect the non-negative entries of an index into a carry array.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_flatten_nextcarry.cpp`.

use crate::kernels::cpu::error::KernelError;

/// Copy every non-negative entry of `fromindex` into `tocarry`, skipping
/// negatives (nulls).  Returns an error if any non-negative entry is `>=
/// lencontent`.
///
/// # Returns
///
/// Number of entries written on success.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_flatten_nextcarry::indexed_array64_flatten_nextcarry_64;
///
/// let from = [0i64, -1, 2, -1, 4];
/// let mut carry = [0i64; 5];
/// let n = indexed_array64_flatten_nextcarry_64(&mut carry, &from, 5).unwrap();
/// assert_eq!(n, 3);
/// assert_eq!(&carry[..n], &[0, 2, 4]);
/// ```
pub fn indexed_array_flatten_nextcarry<C>(
    tocarry: &mut [i64],
    fromindex: &[C],
    lencontent: i64,
) -> Result<usize, KernelError>
where
    C: Copy + Into<i64>,
{
    let mut k = 0usize;
    for (i, &v) in fromindex.iter().enumerate() {
        let j: i64 = v.into();
        if j >= lencontent {
            return Err(KernelError::new("index out of range", i as i64, j));
        } else if j >= 0 {
            tocarry[k] = j;
            k += 1;
        }
    }
    Ok(k)
}

pub fn indexed_array32_flatten_nextcarry_64(
    tocarry: &mut [i64],
    fromindex: &[i32],
    lencontent: i64,
) -> Result<usize, KernelError> {
    indexed_array_flatten_nextcarry(tocarry, fromindex, lencontent)
}
pub fn indexed_array_u32_flatten_nextcarry_64(
    tocarry: &mut [i64],
    fromindex: &[u32],
    lencontent: i64,
) -> Result<usize, KernelError> {
    indexed_array_flatten_nextcarry(tocarry, fromindex, lencontent)
}
pub fn indexed_array64_flatten_nextcarry_64(
    tocarry: &mut [i64],
    fromindex: &[i64],
    lencontent: i64,
) -> Result<usize, KernelError> {
    indexed_array_flatten_nextcarry(tocarry, fromindex, lencontent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_negatives() {
        let from = [0i64, -1, 2, -1, 4];
        let mut carry = [0i64; 5];
        let n = indexed_array64_flatten_nextcarry_64(&mut carry, &from, 5).unwrap();
        assert_eq!(n, 3);
        assert_eq!(&carry[..n], &[0, 2, 4]);
    }

    #[test]
    fn out_of_range_error() {
        let from = [0i64, 99];
        let mut carry = [0i64; 2];
        assert!(indexed_array64_flatten_nextcarry_64(&mut carry, &from, 5).is_err());
    }

    #[test]
    fn empty() {
        let from: [i64; 0] = [];
        let mut carry: [i64; 0] = [];
        let n = indexed_array64_flatten_nextcarry_64(&mut carry, &from, 10).unwrap();
        assert_eq!(n, 0);
    }
}
