// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Compute mask, starts, and stops for a missing-jagged item lookup.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_Content_getitem_next_missing_jagged_getmaskstartstop.cpp`.

/// For each element `i` in `index_in`:
/// * `starts_out[i] = offsets_in[k]`
/// * If `index_in[i] < 0` (missing): `mask_out[i] = -1`, `stops_out[i] = offsets_in[k]`
/// * Otherwise: `mask_out[i] = i`, `k++`, `stops_out[i] = offsets_in[k]`
///
/// # Panics
///
/// Panics if any slice goes out of bounds.
///
/// # Examples
///
/// ```
/// use cpu_kernels::content_getitem_next_missing_jagged_getmaskstartstop::content_getitem_next_missing_jagged_getmaskstartstop;
///
/// let index_in   = [0i64, -1, 1];
/// let offsets_in = [0i64, 3, 7];
/// let mut mask_out  = [0i64; 3];
/// let mut starts    = [0i64; 3];
/// let mut stops     = [0i64; 3];
/// content_getitem_next_missing_jagged_getmaskstartstop(
///     &index_in, &offsets_in, &mut mask_out, &mut starts, &mut stops,
/// );
/// assert_eq!(starts, [0, 3, 3]);
/// assert_eq!(stops,  [3, 3, 7]);
/// assert_eq!(mask_out, [0, -1, 2]);
/// ```
pub fn content_getitem_next_missing_jagged_getmaskstartstop(
    index_in: &[i64],
    offsets_in: &[i64],
    mask_out: &mut [i64],
    starts_out: &mut [i64],
    stops_out: &mut [i64],
) {
    let length = index_in.len();
    assert_eq!(mask_out.len(), length);
    assert_eq!(starts_out.len(), length);
    assert_eq!(stops_out.len(), length);

    let mut k = 0usize;
    for i in 0..length {
        starts_out[i] = offsets_in[k];
        if index_in[i] < 0 {
            mask_out[i] = -1;
            stops_out[i] = offsets_in[k];
        } else {
            mask_out[i] = i as i64;
            k += 1;
            stops_out[i] = offsets_in[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_present() {
        let index_in = [0i64, 1, 2];
        let offsets_in = [0i64, 2, 5, 9];
        let mut mask = [0i64; 3];
        let mut starts = [0i64; 3];
        let mut stops = [0i64; 3];
        content_getitem_next_missing_jagged_getmaskstartstop(
            &index_in,
            &offsets_in,
            &mut mask,
            &mut starts,
            &mut stops,
        );
        assert_eq!(starts, [0, 2, 5]);
        assert_eq!(stops, [2, 5, 9]);
        assert_eq!(mask, [0, 1, 2]);
    }

    #[test]
    fn first_missing() {
        let index_in = [-1i64, 0];
        let offsets_in = [0i64, 4];
        let mut mask = [0i64; 2];
        let mut starts = [0i64; 2];
        let mut stops = [0i64; 2];
        content_getitem_next_missing_jagged_getmaskstartstop(
            &index_in,
            &offsets_in,
            &mut mask,
            &mut starts,
            &mut stops,
        );
        assert_eq!(starts, [0, 0]);
        assert_eq!(stops, [0, 4]);
        assert_eq!(mask, [-1, 1]);
    }

    #[test]
    fn mixed() {
        let index_in = [0i64, -1, 1];
        let offsets_in = [0i64, 3, 7];
        let mut mask = [0i64; 3];
        let mut starts = [0i64; 3];
        let mut stops = [0i64; 3];
        content_getitem_next_missing_jagged_getmaskstartstop(
            &index_in,
            &offsets_in,
            &mut mask,
            &mut starts,
            &mut stops,
        );
        assert_eq!(starts, [0, 3, 3]);
        assert_eq!(stops, [3, 3, 7]);
        assert_eq!(mask, [0, -1, 2]);
    }
}
