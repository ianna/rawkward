// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Copy starts/stops into a destination slice with a base offset added.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_fill.cpp`.

/// Write `tostarts[tostartsoffset+i] = fromstarts[i] + base` and similarly
/// for `tostops`, for each `i`.
pub fn list_array_fill<FROM>(
    tostarts: &mut [i64],
    tostartsoffset: usize,
    tostops: &mut [i64],
    tostopsoffset: usize,
    fromstarts: &[FROM],
    fromstops: &[FROM],
    base: i64,
) where
    FROM: Copy + Into<i64>,
{
    assert_eq!(fromstarts.len(), fromstops.len());
    for i in 0..fromstarts.len() {
        tostarts[tostartsoffset + i] = fromstarts[i].into() + base;
        tostops[tostopsoffset + i] = fromstops[i].into() + base;
    }
}

pub fn list_array_fill_to64_from32(
    tostarts: &mut [i64],
    tso: usize,
    tostops: &mut [i64],
    tpo: usize,
    fromstarts: &[i32],
    fromstops: &[i32],
    base: i64,
) {
    list_array_fill(tostarts, tso, tostops, tpo, fromstarts, fromstops, base);
}
pub fn list_array_fill_to64_fromu32(
    tostarts: &mut [i64],
    tso: usize,
    tostops: &mut [i64],
    tpo: usize,
    fromstarts: &[u32],
    fromstops: &[u32],
    base: i64,
) {
    list_array_fill(tostarts, tso, tostops, tpo, fromstarts, fromstops, base);
}
pub fn list_array_fill_to64_from64(
    tostarts: &mut [i64],
    tso: usize,
    tostops: &mut [i64],
    tpo: usize,
    fromstarts: &[i64],
    fromstops: &[i64],
    base: i64,
) {
    list_array_fill(tostarts, tso, tostops, tpo, fromstarts, fromstops, base);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let fromstarts = [0i64, 3];
        let fromstops = [3i64, 5];
        let mut tostarts = [0i64; 4];
        let mut tostops = [0i64; 4];
        list_array_fill_to64_from64(
            &mut tostarts,
            1,
            &mut tostops,
            1,
            &fromstarts,
            &fromstops,
            10,
        );
        assert_eq!(tostarts, [0, 10, 13, 0]);
        assert_eq!(tostops, [0, 13, 15, 0]);
    }

    #[test]
    fn zero_base() {
        let fromstarts = [0i32, 2];
        let fromstops = [2i32, 4];
        let mut ts = [0i64; 2];
        let mut tp = [0i64; 2];
        list_array_fill_to64_from32(&mut ts, 0, &mut tp, 0, &fromstarts, &fromstops, 0);
        assert_eq!(ts, [0, 2]);
        assert_eq!(tp, [2, 4]);
    }
}
