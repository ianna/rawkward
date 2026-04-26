//! Compose two index arrays into a single flattened index.
//!
//! Corresponds to `src/cpu-kernels/awkward_IndexedArray_simplify.cpp`.

// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

use crate::kernels::cpu::error::KernelError;

/// For each `i`, look up `j = outerindex[i]`:
/// * `j < 0` → `toindex[i] = -1`.
/// * `j >= innerlength` → error.
/// * otherwise → `toindex[i] = innerindex[j]`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::indexed_array_simplify::indexed_array64_simplify64_to64;
///
/// let outer = [0i64, -1, 1];
/// let inner = [10i64, 20];
/// let mut out = [0i64; 3];
/// indexed_array64_simplify64_to64(&mut out, &outer, &inner).unwrap();
/// assert_eq!(out, [10, -1, 20]);
/// ```
pub fn indexed_array_simplify<OUT, IN>(
    toindex: &mut [i64],
    outerindex: &[OUT],
    innerindex: &[IN],
) -> Result<(), KernelError>
where
    OUT: Copy + Into<i64>,
    IN: Copy + Into<i64>,
{
    let innerlength = innerindex.len() as i64;
    for (i, &v) in outerindex.iter().enumerate() {
        let j: i64 = v.into();
        if j < 0 {
            toindex[i] = -1;
        } else if j >= innerlength {
            return Err(KernelError::new("index out of range", i as i64, j));
        } else {
            toindex[i] = innerindex[j as usize].into();
        }
    }
    Ok(())
}

macro_rules! impl_simplify {
    ($fn_name:ident, $outer:ty, $inner:ty) => {
        pub fn $fn_name(
            toindex: &mut [i64],
            outerindex: &[$outer],
            innerindex: &[$inner],
        ) -> Result<(), KernelError> {
            indexed_array_simplify(toindex, outerindex, innerindex)
        }
    };
}

impl_simplify!(indexed_array32_simplify32_to64, i32, i32);
impl_simplify!(indexed_array32_simplify_u32_to64, i32, u32);
impl_simplify!(indexed_array32_simplify64_to64, i32, i64);
impl_simplify!(indexed_array_u32_simplify32_to64, u32, i32);
impl_simplify!(indexed_array_u32_simplify_u32_to64, u32, u32);
impl_simplify!(indexed_array_u32_simplify64_to64, u32, i64);
impl_simplify!(indexed_array64_simplify32_to64, i64, i32);
impl_simplify!(indexed_array64_simplify_u32_to64, i64, u32);
impl_simplify!(indexed_array64_simplify64_to64, i64, i64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let outer = [0i64, -1, 1];
        let inner = [10i64, 20];
        let mut out = [0i64; 3];
        indexed_array64_simplify64_to64(&mut out, &outer, &inner).unwrap();
        assert_eq!(out, [10, -1, 20]);
    }

    #[test]
    fn out_of_range() {
        let outer = [5i64];
        let inner = [10i64, 20];
        let mut out = [0i64; 1];
        assert!(indexed_array64_simplify64_to64(&mut out, &outer, &inner).is_err());
    }

    #[test]
    fn i32_outer() {
        let outer = [1i32, -1, 0];
        let inner = [100i64, 200];
        let mut out = [0i64; 3];
        indexed_array32_simplify64_to64(&mut out, &outer, &inner).unwrap();
        assert_eq!(out, [200, -1, 100]);
    }
}
