// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Collect carry indices and output-index for an indexed array lookup.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_IndexedArray_getitem_nextcarry_outindex.cpp`.

use crate::kernels::cpu::error::KernelError;

/// For each entry `j = fromindex[i]`:
/// * `j >= lencontent` → error.
/// * `j < 0` → `toindex[i] = -1` (null).
/// * `j >= 0` → `tocarry[k] = j`, `toindex[i] = k`, `k++`.
///
/// # Returns
///
/// Number of valid entries written on success.
pub fn indexed_array_getitem_nextcarry_outindex<C>(
    tocarry: &mut [i64],
    toindex: &mut [C],
    fromindex: &[C],
    lencontent: i64,
) -> Result<usize, KernelError>
where
    C: TryFrom<i64> + Copy + Into<i64>,
    <C as TryFrom<i64>>::Error: std::fmt::Debug,
{
    let minus_one = C::try_from(-1i64).expect("-1 fits");
    let mut k = 0usize;
    for (i, &v) in fromindex.iter().enumerate() {
        let j: i64 = v.into();
        if j >= lencontent {
            return Err(KernelError::new("index out of range", i as i64, j));
        } else if j < 0 {
            toindex[i] = minus_one;
        } else {
            tocarry[k] = j;
            toindex[i] = C::try_from(k as i64).expect("k fits");
            k += 1;
        }
    }
    Ok(k)
}

pub fn indexed_array64_getitem_nextcarry_outindex_64(
    tocarry: &mut [i64],
    toindex: &mut [i64],
    fromindex: &[i64],
    lencontent: i64,
) -> Result<usize, KernelError> {
    indexed_array_getitem_nextcarry_outindex(tocarry, toindex, fromindex, lencontent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed() {
        let from = [0i64, -1, 2];
        let mut carry = [0i64; 3];
        let mut idx = [0i64; 3];
        let n =
            indexed_array64_getitem_nextcarry_outindex_64(&mut carry, &mut idx, &from, 5).unwrap();
        assert_eq!(n, 2);
        assert_eq!(&carry[..n], &[0, 2]);
        assert_eq!(idx, [0, -1, 1]);
    }

    #[test]
    fn out_of_range() {
        let from = [99i64];
        let mut carry = [0i64; 1];
        let mut idx = [0i64; 1];
        assert!(
            indexed_array64_getitem_nextcarry_outindex_64(&mut carry, &mut idx, &from, 5).is_err()
        );
    }
}
