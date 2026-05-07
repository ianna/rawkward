// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! ListOffsetArray kernels.

pub mod argsort_strings;
pub mod drop_none_indexes;
pub mod flatten_offsets;
pub mod local_preparenext;

pub mod reduce_local_nextparents;
pub mod reduce_local_outoffsets;
pub mod reduce_nonlocal_maxcount_offsetscopy;
pub mod reduce_nonlocal_nextshifts;
pub mod reduce_nonlocal_nextstarts;
pub mod reduce_nonlocal_outstartsstops;
pub mod reduce_nonlocal_preparenext;

pub mod rpad_and_clip_axis1;
pub mod rpad_axis1;
pub mod rpad_length_axis1;
pub mod to_regular_array;
