// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Right-pad each list to a target length, writing an index and new starts/stops.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_rpad_axis1.cpp`.

/// For each list `i`, produce a padded flat index and new starts/stops:
///
/// * `tostarts[i] = offset`
/// * For `j` in `0..rangeval`: `toindex[offset + j] = fromstarts[i] + j` (copy existing)
/// * For `j` in `rangeval..target`: `toindex[offset + j] = -1` (pad with null)
/// * `offset += max(target, rangeval)`
/// * `tostops[i] = offset`
///
/// The output `toindex` acts as an `IndexedOptionArray` over the original content.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_array_rpad_axis1::list_array64_rpad_axis1_64;
///
/// // Two lists: [0..2] (len 2) and [2..3] (len 1); target=3
/// let fromstarts = [0i64, 2];
/// let fromstops  = [2i64, 3];
/// let mut toindex  = [0i64; 6];
/// let mut tostarts = [0i64; 2];
/// let mut tostops  = [0i64; 2];
/// list_array64_rpad_axis1_64(&mut toindex, &fromstarts, &fromstops, &mut tostarts, &mut tostops, 3);
/// assert_eq!(toindex,  [0, 1, -1, 2, -1, -1]);
/// assert_eq!(tostarts, [0, 3]);
/// assert_eq!(tostops,  [3, 6]);
/// ```
pub fn list_array_rpad_axis1<C>(
    toindex: &mut [i64],
    fromstarts: &[C],
    fromstops: &[C],
    tostarts: &mut [C],
    tostops: &mut [C],
    target: i64,
) where
    C: TryFrom<i64> + Copy + Into<i64>,
    <C as TryFrom<i64>>::Error: std::fmt::Debug,
{
    let length = fromstarts.len();
    assert_eq!(fromstops.len(), length);
    assert_eq!(tostarts.len(), length);
    assert_eq!(tostops.len(), length);

    let mut offset = 0i64;
    for i in 0..length {
        tostarts[i] = C::try_from(offset).expect("offset fits");
        let rangeval: i64 = fromstops[i].into() - fromstarts[i].into();
        let start: i64 = fromstarts[i].into();
        for j in 0..rangeval {
            toindex[(offset + j) as usize] = start + j;
        }
        for j in rangeval..target {
            toindex[(offset + j) as usize] = -1;
        }
        offset = if target > rangeval {
            offset + target
        } else {
            offset + rangeval
        };
        tostops[i] = C::try_from(offset).expect("offset fits");
    }
}

pub fn list_array32_rpad_axis1_64(
    toindex: &mut [i64],
    fromstarts: &[i32],
    fromstops: &[i32],
    tostarts: &mut [i32],
    tostops: &mut [i32],
    target: i64,
) {
    list_array_rpad_axis1(toindex, fromstarts, fromstops, tostarts, tostops, target);
}
pub fn list_array_u32_rpad_axis1_64(
    toindex: &mut [i64],
    fromstarts: &[u32],
    fromstops: &[u32],
    tostarts: &mut [u32],
    tostops: &mut [u32],
    target: i64,
) {
    list_array_rpad_axis1(toindex, fromstarts, fromstops, tostarts, tostops, target);
}
pub fn list_array64_rpad_axis1_64(
    toindex: &mut [i64],
    fromstarts: &[i64],
    fromstops: &[i64],
    tostarts: &mut [i64],
    tostops: &mut [i64],
    target: i64,
) {
    list_array_rpad_axis1(toindex, fromstarts, fromstops, tostarts, tostops, target);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_padding() {
        let fromstarts = [0i64, 2];
        let fromstops = [2i64, 3];
        let mut toindex = [0i64; 6];
        let mut tostarts = [0i64; 2];
        let mut tostops = [0i64; 2];
        list_array64_rpad_axis1_64(
            &mut toindex,
            &fromstarts,
            &fromstops,
            &mut tostarts,
            &mut tostops,
            3,
        );
        assert_eq!(toindex, [0, 1, -1, 2, -1, -1]);
        assert_eq!(tostarts, [0, 3]);
        assert_eq!(tostops, [3, 6]);
    }

    #[test]
    fn no_padding_needed() {
        // Lists already at target length
        let fromstarts = [0i64, 3];
        let fromstops = [3i64, 6];
        let mut toindex = [0i64; 6];
        let mut tostarts = [0i64; 2];
        let mut tostops = [0i64; 2];
        list_array64_rpad_axis1_64(
            &mut toindex,
            &fromstarts,
            &fromstops,
            &mut tostarts,
            &mut tostops,
            3,
        );
        assert_eq!(toindex, [0, 1, 2, 3, 4, 5]);
        assert_eq!(tostarts, [0, 3]);
        assert_eq!(tostops, [3, 6]);
    }

    #[test]
    fn longer_than_target() {
        // List longer than target → no padding, rangeval used
        let fromstarts = [0i64];
        let fromstops = [5i64];
        let mut toindex = [0i64; 5];
        let mut tostarts = [0i64; 1];
        let mut tostops = [0i64; 1];
        list_array64_rpad_axis1_64(
            &mut toindex,
            &fromstarts,
            &fromstops,
            &mut tostarts,
            &mut tostops,
            3,
        );
        assert_eq!(toindex, [0, 1, 2, 3, 4]);
        assert_eq!(tostarts, [0]);
        assert_eq!(tostops, [5]);
    }

    #[test]
    fn i32_variant() {
        let fromstarts = [0i32];
        let fromstops = [2i32];
        let mut toindex = [0i64; 3];
        let mut tostarts = [0i32; 1];
        let mut tostops = [0i32; 1];
        list_array32_rpad_axis1_64(
            &mut toindex,
            &fromstarts,
            &fromstops,
            &mut tostarts,
            &mut tostops,
            3,
        );
        assert_eq!(toindex, [0, 1, -1]);
        assert_eq!(tostarts, [0]);
        assert_eq!(tostops, [3]);
    }
}
