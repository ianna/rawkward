// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Argsort each group of a flat array, writing local (within-group) indices.
//!
//! Corresponds to `src/cpu-kernels/awkward_argsort.cpp`.
//!
//! The comparison follows the C++ NaN-handling convention:
//! NaN sorts *before* any non-NaN value in ascending order (placed first),
//! and *after* any non-NaN value in descending order (placed last).

/// Comparison trait that mirrors the C++ `argsort_order_ascending/descending`
/// logic, including NaN-first semantics for floats.
pub trait ArgsortOrd {
    fn argsort_less(&self, other: &Self) -> bool;
    fn argsort_greater(&self, other: &Self) -> bool;
}

macro_rules! impl_argsort_ord_int {
    ($t:ty) => {
        impl ArgsortOrd for $t {
            fn argsort_less(&self, other: &Self) -> bool {
                self < other
            }
            fn argsort_greater(&self, other: &Self) -> bool {
                self > other
            }
        }
    };
}
impl_argsort_ord_int!(i8);
impl_argsort_ord_int!(u8);
impl_argsort_ord_int!(i16);
impl_argsort_ord_int!(u16);
impl_argsort_ord_int!(i32);
impl_argsort_ord_int!(u32);
impl_argsort_ord_int!(i64);
impl_argsort_ord_int!(u64);
impl_argsort_ord_int!(bool);

// Floats: NaN before non-NaN ascending, NaN after descending.
impl ArgsortOrd for f32 {
    fn argsort_less(&self, other: &Self) -> bool {
        !other.is_nan() && (self.is_nan() || self < other)
    }
    fn argsort_greater(&self, other: &Self) -> bool {
        !other.is_nan() && (self.is_nan() || self > other)
    }
}
impl ArgsortOrd for f64 {
    fn argsort_less(&self, other: &Self) -> bool {
        !other.is_nan() && (self.is_nan() || self < other)
    }
    fn argsort_greater(&self, other: &Self) -> bool {
        !other.is_nan() && (self.is_nan() || self > other)
    }
}

/// Argsort each contiguous group defined by `offsets` and write **local**
/// (0-based within the group) indices into `toptr`.
///
/// `toptr` is initialised to `0, 1, 2, …, length-1` (global), then each
/// segment is sorted in-place, and finally the global indices are made local
/// by subtracting `offsets[i]`.
///
/// # Examples
///
/// ```
/// use cpu_kernels::argsort::argsort_int32;
///
/// // Two groups: [3, 1, 2] and [5, 4]
/// let fromptr = [3i32, 1, 2, 5, 4];
/// let offsets  = [0i64, 3, 5];
/// let mut toptr = [0i64; 5];
/// argsort_int32(&mut toptr, &fromptr, 5, &offsets, 3, true, true);
/// assert_eq!(toptr, [1, 2, 0, 1, 0]);
/// ```
pub fn argsort<T>(
    toptr: &mut [i64],
    fromptr: &[T],
    _length: usize,
    offsets: &[i64],
    offsetslength: usize,
    ascending: bool,
    stable: bool,
) where
    T: ArgsortOrd + Copy,
{
    // Initialise with global indices.
    for (i, v) in toptr.iter_mut().enumerate() {
        *v = i as i64;
    }

    for i in 0..offsetslength.saturating_sub(1) {
        let start = offsets[i] as usize;
        let stop = offsets[i + 1] as usize;
        let seg = &mut toptr[start..stop];

        if stable {
            if ascending {
                seg.sort_by(|&a, &b| {
                    if fromptr[a as usize].argsort_less(&fromptr[b as usize]) {
                        std::cmp::Ordering::Less
                    } else if fromptr[b as usize].argsort_less(&fromptr[a as usize]) {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Equal
                    }
                });
            } else {
                seg.sort_by(|&a, &b| {
                    if fromptr[a as usize].argsort_greater(&fromptr[b as usize]) {
                        std::cmp::Ordering::Less
                    } else if fromptr[b as usize].argsort_greater(&fromptr[a as usize]) {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Equal
                    }
                });
            }
        } else if ascending {
            seg.sort_unstable_by(|&a, &b| {
                if fromptr[a as usize].argsort_less(&fromptr[b as usize]) {
                    std::cmp::Ordering::Less
                } else if fromptr[b as usize].argsort_less(&fromptr[a as usize]) {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            });
        } else {
            seg.sort_unstable_by(|&a, &b| {
                if fromptr[a as usize].argsort_greater(&fromptr[b as usize]) {
                    std::cmp::Ordering::Less
                } else if fromptr[b as usize].argsort_greater(&fromptr[a as usize]) {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            });
        }
        // Make local.
        let base = offsets[i];
        for v in seg.iter_mut() {
            *v -= base;
        }
    }
}

macro_rules! impl_argsort_typed {
    ($fn_name:ident, $t:ty) => {
        pub fn $fn_name(
            toptr: &mut [i64],
            fromptr: &[$t],
            length: usize,
            offsets: &[i64],
            offsetslength: usize,
            ascending: bool,
            stable: bool,
        ) {
            argsort(
                toptr,
                fromptr,
                length,
                offsets,
                offsetslength,
                ascending,
                stable,
            )
        }
    };
}
impl_argsort_typed!(argsort_bool, bool);
impl_argsort_typed!(argsort_int8, i8);
impl_argsort_typed!(argsort_uint8, u8);
impl_argsort_typed!(argsort_int16, i16);
impl_argsort_typed!(argsort_uint16, u16);
impl_argsort_typed!(argsort_int32, i32);
impl_argsort_typed!(argsort_uint32, u32);
impl_argsort_typed!(argsort_int64, i64);
impl_argsort_typed!(argsort_uint64, u64);
impl_argsort_typed!(argsort_float32, f32);
impl_argsort_typed!(argsort_float64, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascending_stable() {
        let from = [3i32, 1, 2, 5, 4];
        let offsets = [0i64, 3, 5];
        let mut out = [0i64; 5];
        argsort_int32(&mut out, &from, 5, &offsets, 3, true, true);
        assert_eq!(out, [1, 2, 0, 1, 0]);
    }

    #[test]
    fn descending() {
        let from = [1i32, 3, 2];
        let offsets = [0i64, 3];
        let mut out = [0i64; 3];
        argsort_int32(&mut out, &from, 3, &offsets, 2, false, true);
        assert_eq!(out, [1, 2, 0]);
    }

    #[test]
    fn float_nan_ascending() {
        // NaN sorts first in ascending
        let from = [2.0f64, f64::NAN, 1.0];
        let offsets = [0i64, 3];
        let mut out = [0i64; 3];
        argsort_float64(&mut out, &from, 3, &offsets, 2, true, true);
        assert_eq!(out[0], 1); // NaN first
    }

    #[test]
    fn single_group_already_sorted() {
        let from = [1i64, 2, 3];
        let offsets = [0i64, 3];
        let mut out = [0i64; 3];
        argsort_int64(&mut out, &from, 3, &offsets, 2, true, true);
        assert_eq!(out, [0, 1, 2]);
    }
}
