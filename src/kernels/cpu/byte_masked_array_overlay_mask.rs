// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Overlay two byte masks with a logical OR.
//!
//! Corresponds to `src/cpu-kernels/awkward_ByteMaskedArray_overlay_mask.cpp`.

/// Combine `mymask` and `theirmask` into `tomask` using a logical OR, where
/// "null" is encoded as a non-zero byte.
///
/// For each position `i`:
/// ```text
/// their_null = theirmask[i] != 0                        // non-zero encodes null
/// my_null    = (mymask[i] != 0) != validwhen            // apply validwhen only to mymask
/// tomask[i]  = (their_null || my_null) as i8            // 1 = null, 0 = valid
/// ```
///
/// The semantics match those of `ByteMaskedArray.overlay_mask` in the
/// awkward-array Python library.
///
/// # Parameters
///
/// * `tomask`    – Output mask; same length as `mymask` and `theirmask`.
/// * `theirmask` – Base mask; non-zero = null (no `validwhen` applied).
/// * `mymask`    – Overlay mask; interpreted via `validwhen`.
/// * `validwhen` – The byte truth-value that signals a *valid* element in `mymask`.
///
/// # Panics
///
/// Panics if the three slices have different lengths.
///
/// # Examples
///
/// ```
/// use cpu_kernels::byte_masked_array_overlay_mask::byte_masked_array_overlay_mask_8;
///
/// let theirmask = [0i8, 1, 0, 0]; // position 1 already null
/// let mymask    = [0i8, 0, 1, 0]; // validwhen=true: position 2 is null (1 != true → null)... wait
/// // With validwhen=true: my_null[i] = (mymask[i]!=0) != true
/// //   i=0: false!=true=true  → null → but their=0 too → 1? No...
/// // validwhen=true means mymask[i]=1 encodes VALID (matches validwhen),
/// // so my_null = (mymask[i]!=0) != validwhen:
/// //   mymask[0]=0: false!=true = true  → null
/// //   mymask[2]=1: true!=true  = false → valid
/// let mut tomask = [0i8; 4];
/// byte_masked_array_overlay_mask_8(&mut tomask, &theirmask, &mymask, true);
/// // i=0: their=0(valid), my_null=true  → 1
/// // i=1: their=1(null),  my_null=true  → 1
/// // i=2: their=0(valid), my_null=false → 0
/// // i=3: their=0(valid), my_null=true  → 1
/// assert_eq!(tomask, [1, 1, 0, 1]);
/// ```
pub fn byte_masked_array_overlay_mask(
    tomask: &mut [i8],
    theirmask: &[i8],
    mymask: &[i8],
    validwhen: bool,
) {
    assert_eq!(tomask.len(), theirmask.len());
    assert_eq!(tomask.len(), mymask.len());

    for i in 0..tomask.len() {
        let their_null = theirmask[i] != 0;
        let my_null = (mymask[i] != 0) != validwhen;
        tomask[i] = (their_null || my_null) as i8;
    }
}

/// Typed wrapper for `i8` masks (mirrors `awkward_ByteMaskedArray_overlay_mask8`).
pub fn byte_masked_array_overlay_mask_8(
    tomask: &mut [i8],
    theirmask: &[i8],
    mymask: &[i8],
    validwhen: bool,
) {
    byte_masked_array_overlay_mask(tomask, theirmask, mymask, validwhen);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Encoding used throughout:
    //   theirmask: non-zero = null (raw; validwhen not applied)
    //   mymask:    (value != 0) != validwhen == true  →  null

    #[test]
    fn no_null_anywhere() {
        // theirmask all 0 (valid), mymask all 1 with validwhen=true (1==true → valid).
        // my_null = (1!=0)!=true = false → valid everywhere.
        let theirmask = [0i8; 4];
        let mymask = [1i8; 4];
        let mut tomask = [0i8; 4];
        byte_masked_array_overlay_mask_8(&mut tomask, &theirmask, &mymask, true);
        assert_eq!(tomask, [0, 0, 0, 0]);
    }

    #[test]
    fn their_mask_dominates() {
        // theirmask all 1 → their_null=true everywhere → all null regardless of mymask.
        let theirmask = [1i8; 4];
        let mymask = [1i8; 4];
        let mut tomask = [0i8; 4];
        byte_masked_array_overlay_mask_8(&mut tomask, &theirmask, &mymask, true);
        assert_eq!(tomask, [1, 1, 1, 1]);
    }

    #[test]
    fn my_mask_adds_nulls() {
        // theirmask all 0 (valid). mymask all 0 with validwhen=false:
        // my_null = (0!=0)!=false = false!=false = false → valid.
        let theirmask = [0i8; 4];
        let mymask = [0i8; 4];
        let mut tomask = [0i8; 4];
        byte_masked_array_overlay_mask_8(&mut tomask, &theirmask, &mymask, false);
        assert_eq!(tomask, [0, 0, 0, 0]);
    }

    #[test]
    fn my_mask_adds_nulls_nonzero() {
        // theirmask all 0. mymask all 1 with validwhen=false:
        // my_null = (1!=0)!=false = true!=false = true → all null.
        let theirmask = [0i8; 4];
        let mymask = [1i8; 4];
        let mut tomask = [0i8; 4];
        byte_masked_array_overlay_mask_8(&mut tomask, &theirmask, &mymask, false);
        assert_eq!(tomask, [1, 1, 1, 1]);
    }

    #[test]
    fn mixed_overlay() {
        // theirmask=[0,1,0,0], mymask=[0,0,1,0], validwhen=true.
        // their_null: [F, T, F, F]
        // my_null = (v!=0)!=true: [T, T, F, T]  (0→false→true, 1→true→false)
        // OR:       [T, T, F, T] → [1,1,0,1]
        let theirmask = [0i8, 1, 0, 0];
        let mymask = [0i8, 0, 1, 0];
        let mut tomask = [0i8; 4];
        byte_masked_array_overlay_mask_8(&mut tomask, &theirmask, &mymask, true);
        assert_eq!(tomask, [1, 1, 0, 1]);
    }

    #[test]
    fn my_null_overlaid() {
        // theirmask all 0. mymask[0]=1, validwhen=false → my_null[0]=true → null.
        let theirmask = [0i8; 4];
        let mymask = [1i8, 0, 0, 0];
        let mut tomask = [0i8; 4];
        byte_masked_array_overlay_mask_8(&mut tomask, &theirmask, &mymask, false);
        assert_eq!(tomask, [1, 0, 0, 0]);
    }
}
