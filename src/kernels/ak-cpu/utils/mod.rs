// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

/// Clamp and normalise a `(start, stop)` range slice against an array of
/// `length` elements, applying Python-style negative-index semantics.
///
/// # Parameters
///
/// * `start` – Inclusive start of the slice; `None` means "use the default".
/// * `stop`  – Exclusive stop of the slice; `None` means "use the default".
/// * `posstep` – `true` if the step is positive (forward iteration).
/// * `length` – Length of the array being sliced.
///
/// # Returns
///
/// A `(start, stop)` pair clamped to `[0, length]` for positive steps, or to
/// `[-1, length-1]` (inclusive both ends) for negative steps.
///
/// # Examples
///
/// ```
/// use cpu_kernels::kernel_utils::regularize_rangeslice;
///
/// // Forward slice with defaults: equivalent to [0..5]
/// assert_eq!(regularize_rangeslice(None, None, true, 5), (0, 5));
///
/// // Negative start index
/// assert_eq!(regularize_rangeslice(Some(-2), None, true, 5), (3, 5));
///
/// // Reverse slice
/// assert_eq!(regularize_rangeslice(None, None, false, 5), (4, -1));
/// ```
pub fn regularize_rangeslice(
    start: Option<i64>,
    stop: Option<i64>,
    posstep: bool,
    length: i64,
) -> (i64, i64) {
    if posstep {
        let mut s = match start {
            None => 0,
            Some(v) if v < 0 => (v + length).max(0),
            Some(v) => v.min(length),
        };
        s = s.clamp(0, length);

        let mut e = match stop {
            None => length,
            Some(v) if v < 0 => (v + length).max(0),
            Some(v) => v.min(length),
        };
        e = e.clamp(0, length);
        if e < s {
            e = s;
        }

        (s, e)
    } else {
        let mut s = match start {
            None => length - 1,
            Some(v) if v < 0 => (v + length).max(-1),
            Some(v) => v.min(length - 1),
        };
        s = s.clamp(-1, length - 1);

        let mut e = match stop {
            None => -1,
            Some(v) if v < 0 => (v + length).max(-1),
            Some(v) => v.min(length - 1),
        };
        e = e.clamp(-1, length - 1);
        if e > s {
            e = s;
        }

        (s, e)
    }
}

/// Recursive helper that enumerates all combinations of `n` indices drawn from
/// `[fromindex[j], stop)`, writing the results into `tocarry`.
///
/// This is a direct translation of the generic `awkward_ListArray_combinations_step`
/// template from C++.  It is intentionally not `pub` – callers should use the
/// typed wrappers below.
fn combinations_step<T>(
    tocarry: &mut [Vec<T>],
    toindex: &mut [usize],
    fromindex: &mut Vec<i64>,
    j: usize,
    stop: i64,
    n: usize,
    replacement: bool,
) where
    T: TryFrom<i64> + Copy,
    <T as TryFrom<i64>>::Error: std::fmt::Debug,
{
    while fromindex[j] < stop {
        // Propagate the current index forward to later positions.
        if replacement {
            for k in (j + 1)..n {
                fromindex[k] = fromindex[j];
            }
        } else {
            for k in (j + 1)..n {
                fromindex[k] = fromindex[j] + (k - j) as i64;
            }
        }

        if j + 1 == n {
            // Leaf: record one combination.
            for k in 0..n {
                tocarry[k][toindex[k]] = T::try_from(fromindex[k]).expect("index fits in T");
                toindex[k] += 1;
            }
        } else {
            combinations_step(tocarry, toindex, fromindex, j + 1, stop, n, replacement);
        }

        fromindex[j] += 1;
    }
}

/// Generate all `n`-combinations (or combinations with replacement) of
/// indices in `[start, stop)`, writing each "column" into the corresponding
/// `Vec<i64>` in `tocarry`.
///
/// Mirrors `awkward_ListArray_combinations_step_64`.
///
/// # Panics
///
/// Panics if `tocarry.len() != n` or if any `tocarry[k]` is too short to
/// hold all combinations.
pub fn list_array_combinations_step_64(
    tocarry: &mut [Vec<i64>],
    toindex: &mut [usize],
    fromindex: &mut Vec<i64>,
    j: usize,
    stop: i64,
    n: usize,
    replacement: bool,
) {
    combinations_step(tocarry, toindex, fromindex, j, stop, n, replacement);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rangeslice_forward_defaults() {
        assert_eq!(regularize_rangeslice(None, None, true, 5), (0, 5));
    }

    #[test]
    fn rangeslice_forward_negative_start() {
        assert_eq!(regularize_rangeslice(Some(-2), None, true, 5), (3, 5));
    }

    #[test]
    fn rangeslice_forward_out_of_bounds() {
        assert_eq!(regularize_rangeslice(Some(10), None, true, 5), (5, 5));
    }

    #[test]
    fn rangeslice_forward_stop_before_start() {
        assert_eq!(regularize_rangeslice(Some(3), Some(1), true, 5), (3, 3));
    }

    #[test]
    fn rangeslice_reverse_defaults() {
        assert_eq!(regularize_rangeslice(None, None, false, 5), (4, -1));
    }

    #[test]
    fn rangeslice_reverse_negative_start() {
        assert_eq!(regularize_rangeslice(Some(-1), None, false, 5), (4, -1));
    }

    #[test]
    fn combinations_step_pairs() {
        // 2-combinations of [0,1,2] → (0,1),(0,2),(1,2)
        let n = 2;
        let mut tocarry: Vec<Vec<i64>> = vec![vec![0i64; 3], vec![0i64; 3]];
        let mut toindex = vec![0usize; n];
        let mut fromindex = vec![0i64, 1i64];
        list_array_combinations_step_64(&mut tocarry, &mut toindex, &mut fromindex, 0, 3, n, false);
        assert_eq!(tocarry[0][..toindex[0]], [0, 0, 1]);
        assert_eq!(tocarry[1][..toindex[1]], [1, 2, 2]);
    }
}
