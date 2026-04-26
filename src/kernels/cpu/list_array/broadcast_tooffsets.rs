// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Broadcast a ListArray to a target offsets array, producing a carry index.
//!
//! Corresponds to `src/cpu-kernels/awkward_ListArray_broadcast_tooffsets.cpp`.

use crate::kernels::cpu::error::KernelError;

/// Verify that list `i` can be broadcast to `fromoffsets[i+1]-fromoffsets[i]`
/// elements, then fill `tocarry` with positions `start..stop`.
///
/// # Errors
///
/// * `stops[i] > lencontent` – list overflows content.
/// * `fromoffsets` not monotonically increasing.
/// * List length doesn't match broadcast length.
pub fn list_array_broadcast_tooffsets<C>(
    tocarry: &mut [i64],
    fromoffsets: &[i64],
    fromstarts: &[C],
    fromstops: &[C],
    lencontent: i64,
) -> Result<(), KernelError>
where
    C: Copy + Into<i64>,
{
    let nlists = fromoffsets.len().saturating_sub(1);
    let mut k = 0usize;
    for i in 0..nlists {
        let start: i64 = fromstarts[i].into();
        let stop: i64 = fromstops[i].into();
        if start != stop && stop > lencontent {
            return Err(KernelError::new("stops[i] > len(content)", i as i64, stop));
        }
        let count = fromoffsets[i + 1] - fromoffsets[i];
        if count < 0 {
            return Err(KernelError::at(
                "broadcast's offsets must be monotonically increasing",
                i as i64,
            ));
        }
        if stop - start != count {
            return Err(KernelError::at("cannot broadcast nested list", i as i64));
        }
        for j in start..stop {
            tocarry[k] = j;
            k += 1;
        }
    }
    Ok(())
}

pub fn list_array32_broadcast_tooffsets_64(
    tocarry: &mut [i64],
    fromoffsets: &[i64],
    fromstarts: &[i32],
    fromstops: &[i32],
    lencontent: i64,
) -> Result<(), KernelError> {
    list_array_broadcast_tooffsets(tocarry, fromoffsets, fromstarts, fromstops, lencontent)
}
pub fn list_array_u32_broadcast_tooffsets_64(
    tocarry: &mut [i64],
    fromoffsets: &[i64],
    fromstarts: &[u32],
    fromstops: &[u32],
    lencontent: i64,
) -> Result<(), KernelError> {
    list_array_broadcast_tooffsets(tocarry, fromoffsets, fromstarts, fromstops, lencontent)
}
pub fn list_array64_broadcast_tooffsets_64(
    tocarry: &mut [i64],
    fromoffsets: &[i64],
    fromstarts: &[i64],
    fromstops: &[i64],
    lencontent: i64,
) -> Result<(), KernelError> {
    list_array_broadcast_tooffsets(tocarry, fromoffsets, fromstarts, fromstops, lencontent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let offsets = [0i64, 2, 4];
        let starts = [0i64, 2];
        let stops = [2i64, 4];
        let mut carry = [0i64; 4];
        list_array64_broadcast_tooffsets_64(&mut carry, &offsets, &starts, &stops, 4).unwrap();
        assert_eq!(carry, [0, 1, 2, 3]);
    }

    #[test]
    fn empty_list_ok() {
        let offsets = [0i64, 0, 3];
        let starts = [2i64, 0]; // empty first list
        let stops = [2i64, 3];
        let mut carry = [0i64; 3];
        list_array64_broadcast_tooffsets_64(&mut carry, &offsets, &starts, &stops, 5).unwrap();
        assert_eq!(&carry[..3], &[0, 1, 2]);
    }

    #[test]
    fn stop_beyond_content() {
        let offsets = [0i64, 2];
        let starts = [0i64];
        let stops = [99i64];
        let mut carry = [0i64; 99];
        assert!(
            list_array64_broadcast_tooffsets_64(&mut carry, &offsets, &starts, &stops, 5).is_err()
        );
    }

    #[test]
    fn length_mismatch() {
        let offsets = [0i64, 3]; // expects 3 elements
        let starts = [0i64];
        let stops = [2i64]; // but list only has 2
        let mut carry = [0i64; 10];
        assert!(
            list_array64_broadcast_tooffsets_64(&mut carry, &offsets, &starts, &stops, 5).is_err()
        );
    }
}
