// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Copy an index array into a slice of a destination array.
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_fillindex.cpp`.

/// Write `toindex[toindexoffset + i] = fromindex[i]` for each `i`.
pub fn union_array_fillindex<FROM, TO>(toindex: &mut [TO], toindexoffset: usize, fromindex: &[FROM])
where
    FROM: Copy + Into<i64>,
    TO: TryFrom<i64> + Copy,
    <TO as TryFrom<i64>>::Error: std::fmt::Debug,
{
    for (i, &v) in fromindex.iter().enumerate() {
        toindex[toindexoffset + i] = TO::try_from(v.into()).expect("value fits");
    }
}

pub fn union_array_fillindex_to64_from32(toindex: &mut [i64], offset: usize, fromindex: &[i32]) {
    union_array_fillindex(toindex, offset, fromindex);
}
pub fn union_array_fillindex_to64_fromu32(toindex: &mut [i64], offset: usize, fromindex: &[u32]) {
    union_array_fillindex(toindex, offset, fromindex);
}
pub fn union_array_fillindex_to64_from64(toindex: &mut [i64], offset: usize, fromindex: &[i64]) {
    union_array_fillindex(toindex, offset, fromindex);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_copy() {
        let from = [10i64, 20, 30];
        let mut to = [0i64; 6];
        union_array_fillindex_to64_from64(&mut to, 2, &from);
        assert_eq!(to, [0, 0, 10, 20, 30, 0]);
    }

    #[test]
    fn i32_to_i64() {
        let from = [1i32, 2, 3];
        let mut to = [0i64; 3];
        union_array_fillindex_to64_from32(&mut to, 0, &from);
        assert_eq!(to, [1, 2, 3]);
    }

    #[test]
    fn zero_offset() {
        let from = [5u32, 6];
        let mut to = [0i64; 2];
        union_array_fillindex_to64_fromu32(&mut to, 0, &from);
        assert_eq!(to, [5, 6]);
    }
}
