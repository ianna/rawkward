// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Replace nulls (negative values) with `0` in a union index.
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_fillna.cpp`.

/// For each `i`: `toindex[i] = fromindex[i] >= 0 ? fromindex[i] : 0`.
pub fn union_array_fillna<C>(toindex: &mut [i64], fromindex: &[C])
where
    C: Copy + Into<i64>,
{
    assert_eq!(toindex.len(), fromindex.len());
    for (out, &v) in toindex.iter_mut().zip(fromindex.iter()) {
        let j: i64 = v.into();
        *out = if j >= 0 { j } else { 0 };
    }
}

pub fn union_array_fillna_from32_to64(toindex: &mut [i64], fromindex: &[i32]) {
    union_array_fillna(toindex, fromindex);
}
pub fn union_array_fillna_fromu32_to64(toindex: &mut [i64], fromindex: &[u32]) {
    union_array_fillna(toindex, fromindex);
}
pub fn union_array_fillna_from64_to64(toindex: &mut [i64], fromindex: &[i64]) {
    union_array_fillna(toindex, fromindex);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_negatives() {
        let from = [3i64, -1, 5, -2];
        let mut to = [0i64; 4];
        union_array_fillna_from64_to64(&mut to, &from);
        assert_eq!(to, [3, 0, 5, 0]);
    }

    #[test]
    fn no_negatives() {
        let from = [0i32, 1, 2];
        let mut to = [0i64; 3];
        union_array_fillna_from32_to64(&mut to, &from);
        assert_eq!(to, [0, 1, 2]);
    }

    #[test]
    fn all_negative() {
        let from = [-1i64; 4];
        let mut to = [99i64; 4];
        union_array_fillna_from64_to64(&mut to, &from);
        assert_eq!(to, [0, 0, 0, 0]);
    }
}
