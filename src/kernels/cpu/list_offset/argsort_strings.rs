// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Argsort a flat string-array grouped by parent, writing carry indices.
//!
//! Corresponds to
//! `src/cpu-kernels/awkward_ListOffsetArray_argsort_strings.cpp`.

/// Argsort a sequence of strings (represented by `stringstarts`/`stringstops`
/// offsets into `stringdata`) grouped by `fromparents`.
///
/// Within each contiguous run sharing the same parent value, the positions are
/// sorted by the string comparison and written into `tocarry`.  When
/// `is_local`, the written values are local (0-based within the group);
/// otherwise they are the original global indices.
///
/// # Parameters
///
/// * `tocarry`      – Output carry; same length as `fromparents`.
/// * `fromparents`  – Parent-group index for each string; must be
///                    non-decreasing.
/// * `stringdata`   – Raw UTF-8 (or ASCII) bytes of all strings concatenated.
/// * `stringstarts` – Start byte offset of each string in `stringdata`.
/// * `stringstops`  – Exclusive stop byte offset of each string.
/// * `is_stable`    – Use stable sort (preserves relative order of equal strings).
/// * `is_ascending` – Sort ascending; `false` = descending.
/// * `is_local`     – Write group-local indices (0, 1, 2…) rather than global ones.
///
/// # Examples
///
/// ```
/// use cpu_kernels::list_offset_array_argsort_strings::list_offset_array_argsort_strings;
///
/// // Three strings: "banana", "apple", "cherry"
/// let data      = b"bananaaapplecherry";
/// let starts    = [0i64, 6, 11];
/// let stops     = [6i64, 11, 17];
/// let parents   = [0i64, 0, 0];
/// let mut carry = [0i64; 3];
/// list_offset_array_argsort_strings(&mut carry, &parents, data, &starts, &stops, true, true, true);
/// // ascending: "apple" < "banana" < "cherry" → local indices [1, 0, 2]
/// assert_eq!(carry, [1, 0, 2]);
/// ```
pub fn list_offset_array_argsort_strings(
    tocarry: &mut [i64],
    fromparents: &[i64],
    stringdata: &[u8],
    stringstarts: &[i64],
    stringstops: &[i64],
    is_stable: bool,
    is_ascending: bool,
    is_local: bool,
) {
    let length = fromparents.len();
    assert_eq!(tocarry.len(), length);
    assert_eq!(stringstarts.len(), length);
    assert_eq!(stringstops.len(), length);

    let cmp_strings = |left: usize, right: usize| -> std::cmp::Ordering {
        let ls = stringstarts[left] as usize;
        let le = stringstops[left] as usize;
        let rs = stringstarts[right] as usize;
        let re = stringstops[right] as usize;
        let l_str = &stringdata[ls..le];
        let r_str = &stringdata[rs..re];
        l_str.cmp(r_str)
    };

    let mut first_index = 0usize;
    let mut last_parent = -1i64;
    let mut group: Vec<usize> = Vec::new();

    // Flush helper: sort `group` and write results into `tocarry`.
    let flush = |group: &mut Vec<usize>,
                 first_index: usize,
                 tocarry: &mut [i64],
                 is_stable: bool,
                 is_ascending: bool,
                 is_local: bool,
                 cmp: &dyn Fn(usize, usize) -> std::cmp::Ordering| {
        if is_stable {
            if is_ascending {
                group.sort_by(|&a, &b| cmp(a, b));
            } else {
                group.sort_by(|&a, &b| cmp(b, a));
            }
        } else if is_ascending {
            group.sort_unstable_by(|&a, &b| cmp(a, b));
        } else {
            group.sort_unstable_by(|&a, &b| cmp(b, a));
        }
        for (j, &global_idx) in group.iter().enumerate() {
            tocarry[first_index + j] = if is_local {
                (global_idx - first_index) as i64
            } else {
                global_idx as i64
            };
        }
        group.clear();
    };

    for i in 0..=length {
        let new_parent = if i < length { fromparents[i] } else { -2 };
        if i == length || new_parent != last_parent {
            if !group.is_empty() {
                flush(
                    &mut group,
                    first_index,
                    tocarry,
                    is_stable,
                    is_ascending,
                    is_local,
                    &cmp_strings,
                );
            }
        }
        if i < length {
            if group.is_empty() {
                first_index = i;
            }
            group.push(i);
            last_parent = new_parent;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_string_data(strings: &[&str]) -> (Vec<u8>, Vec<i64>, Vec<i64>) {
        let mut data = Vec::new();
        let mut starts = Vec::new();
        let mut stops = Vec::new();
        for s in strings {
            starts.push(data.len() as i64);
            data.extend_from_slice(s.as_bytes());
            stops.push(data.len() as i64);
        }
        (data, starts, stops)
    }

    #[test]
    fn ascending_local_single_group() {
        let (data, starts, stops) = make_string_data(&["banana", "apple", "cherry"]);
        let parents = [0i64, 0, 0];
        let mut carry = [0i64; 3];
        list_offset_array_argsort_strings(
            &mut carry, &parents, &data, &starts, &stops, true, true, true,
        );
        assert_eq!(carry, [1, 0, 2]); // "apple"<"banana"<"cherry"
    }

    #[test]
    fn descending_local() {
        let (data, starts, stops) = make_string_data(&["a", "c", "b"]);
        let parents = [0i64, 0, 0];
        let mut carry = [0i64; 3];
        list_offset_array_argsort_strings(
            &mut carry, &parents, &data, &starts, &stops, true, false, true,
        );
        assert_eq!(carry, [1, 2, 0]); // "c">"b">"a"
    }

    #[test]
    fn global_indices() {
        let (data, starts, stops) = make_string_data(&["z", "a"]);
        let parents = [0i64, 0];
        let mut carry = [0i64; 2];
        list_offset_array_argsort_strings(
            &mut carry, &parents, &data, &starts, &stops, true, true, false,
        );
        assert_eq!(carry, [1, 0]); // "a" at global 1, "z" at global 0
    }

    #[test]
    fn two_separate_groups() {
        let (data, starts, stops) = make_string_data(&["b", "a", "d", "c"]);
        let parents = [0i64, 0, 1, 1];
        let mut carry = [0i64; 4];
        list_offset_array_argsort_strings(
            &mut carry, &parents, &data, &starts, &stops, true, true, true,
        );
        // group0: "a"<"b" → [1,0]; group1: "c"<"d" → [1,0]
        assert_eq!(carry, [1, 0, 1, 0]);
    }

    #[test]
    fn stable_preserves_equal_order() {
        let (data, starts, stops) = make_string_data(&["x", "x", "x"]);
        let parents = [0i64, 0, 0];
        let mut carry = [0i64; 3];
        list_offset_array_argsort_strings(
            &mut carry, &parents, &data, &starts, &stops, true, true, true,
        );
        // All equal – stable sort preserves original order
        assert_eq!(carry, [0, 1, 2]);
    }
}
