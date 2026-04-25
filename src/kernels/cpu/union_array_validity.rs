// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Validate the tags and index arrays of a `UnionArray`.
//!
//! Corresponds to `src/cpu-kernels/awkward_UnionArray_validity.cpp`.

use crate::kernels::cpu::error::KernelError;

/// Validate a `UnionArray`'s `tags` and `index` arrays against a list of
/// per-content lengths.
///
/// For each element `i`:
/// 1. `tags[i] >= 0`
/// 2. `index[i] >= 0`
/// 3. `tags[i] < numcontents`
/// 4. `index[i] < lencontents[tags[i]]`
///
/// # Parameters
///
/// * `tags`        – Discriminant array identifying which content each element
///                   belongs to.
/// * `index`       – Per-content position index.
/// * `lencontents` – Length of each content array; indexed by `tags[i]`.
///
/// # Errors
///
/// Returns a [`KernelError`] at the first offending element.
///
/// # Examples
///
/// ```
/// use cpu_kernels::union_array_validity::union_array_validity_8_64;
///
/// let tags        = [0i8, 1, 0];
/// let index       = [0i64, 0, 1];
/// let lencontents = [2i64, 3];   // content 0 has 2 elements, content 1 has 3
/// assert!(union_array_validity_8_64(&tags, &index, &lencontents).is_ok());
/// ```
pub fn union_array_validity<T, I>(
    tags: &[T],
    index: &[I],
    lencontents: &[i64],
) -> Result<(), KernelError>
where
    T: PartialOrd + Copy + Into<i64>,
    I: PartialOrd + Copy + Into<i64>,
{
    assert_eq!(
        tags.len(),
        index.len(),
        "tags and index must have the same length"
    );

    let numcontents = lencontents.len() as i64;

    for i in 0..tags.len() {
        let tag: i64 = tags[i].into();
        let idx: i64 = index[i].into();

        if tag < 0 {
            return Err(KernelError::at("tags[i] < 0", i as i64));
        }
        if idx < 0 {
            return Err(KernelError::at("index[i] < 0", i as i64));
        }
        if tag >= numcontents {
            return Err(KernelError::at("tags[i] >= len(contents)", i as i64));
        }
        let lencontent = lencontents[tag as usize];
        if idx >= lencontent {
            return Err(KernelError::at(
                "index[i] >= len(content[tags[i]])",
                i as i64,
            ));
        }
    }
    Ok(())
}

/// Typed wrapper for `(i8 tags, i32 index)`
/// (mirrors `awkward_UnionArray8_32_validity`).
pub fn union_array_validity_8_32(
    tags: &[i8],
    index: &[i32],
    lencontents: &[i64],
) -> Result<(), KernelError> {
    union_array_validity(tags, index, lencontents)
}

/// Typed wrapper for `(i8 tags, u32 index)`
/// (mirrors `awkward_UnionArray8_U32_validity`).
pub fn union_array_validity_8_u32(
    tags: &[i8],
    index: &[u32],
    lencontents: &[i64],
) -> Result<(), KernelError> {
    union_array_validity(tags, index, lencontents)
}

/// Typed wrapper for `(i8 tags, i64 index)`
/// (mirrors `awkward_UnionArray8_64_validity`).
pub fn union_array_validity_8_64(
    tags: &[i8],
    index: &[i64],
    lencontents: &[i64],
) -> Result<(), KernelError> {
    union_array_validity(tags, index, lencontents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_basic() {
        let tags = [0i8, 1, 0];
        let index = [0i64, 0, 1];
        let lencontents = [2i64, 3];
        assert!(union_array_validity_8_64(&tags, &index, &lencontents).is_ok());
    }

    #[test]
    fn tag_negative() {
        let tags = [-1i8];
        let index = [0i64];
        let lens = [2i64];
        let err = union_array_validity_8_64(&tags, &index, &lens).unwrap_err();
        assert!(err.message.contains("tags[i] < 0"));
    }

    #[test]
    fn index_negative() {
        let tags = [0i8];
        let index = [-1i64];
        let lens = [2i64];
        let err = union_array_validity_8_64(&tags, &index, &lens).unwrap_err();
        assert!(err.message.contains("index[i] < 0"));
    }

    #[test]
    fn tag_out_of_range() {
        let tags = [2i8];
        let index = [0i64];
        let lens = [2i64, 2i64]; // only 2 contents
        let err = union_array_validity_8_64(&tags, &index, &lens).unwrap_err();
        assert!(err.message.contains("tags[i] >= len(contents)"));
    }

    #[test]
    fn index_out_of_content_range() {
        let tags = [0i8];
        let index = [5i64];
        let lens = [3i64];
        let err = union_array_validity_8_64(&tags, &index, &lens).unwrap_err();
        assert!(err.message.contains("index[i] >= len(content[tags[i]])"));
    }

    #[test]
    fn empty_arrays_valid() {
        let tags: [i8; 0] = [];
        let index: [i64; 0] = [];
        assert!(union_array_validity_8_64(&tags, &index, &[]).is_ok());
    }

    #[test]
    fn error_at_second_element() {
        let tags = [0i8, 0];
        let index = [0i64, 99];
        let lens = [2i64];
        let err = union_array_validity_8_64(&tags, &index, &lens).unwrap_err();
        assert_eq!(err.id, 1);
    }

    #[test]
    fn u32_index_valid() {
        let tags = [0i8, 1];
        let index = [0u32, 1];
        let lens = [2i64, 3];
        assert!(union_array_validity_8_u32(&tags, &index, &lens).is_ok());
    }
}
