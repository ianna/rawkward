# CPU kernels layout proposal

A coherent end-state for `src/kernels/cpu/`, derived directly from the
~150 C++ files in `awkward/src/cpu-kernels/`. Designed so:

- **Every C++ file maps to one Rust path.** No ambiguity about where to look.
- **The Rust path makes the array type a directory, not a filename prefix.**
  `awkward_ListArray_getitem_jagged_apply.cpp` → `list_array/getitem/jagged_apply.rs`.
- **The same skeleton mirrors across `cpu/`, `simd/`, `cuda/`, `hip/`** — when
  you add a SIMD or GPU variant of a kernel, its path is mechanically derived
  from the CPU path.
- **Generic core + typed wrappers in a single file**, the way you already do
  it in `reduce_sum.rs` and `reduce_max.rs`. One function = one file is the
  default; only fuse files when the kernels share non-trivial machinery.

---

## 1. Top-level shape

Pull cross-cutting infrastructure (`error`, traits, utils) up out of `cpu/`
so the SIMD/CUDA/HIP backends can share it.

```
src/kernels/
├── mod.rs               // re-exports + `pub mod cpu; pub mod simd; …`
├── error.rs             // KernelError, IndexedError — shared by all backends
├── traits.rs            // ArgsortOrd, One, ToDType-like helpers
├── utils.rs             // regularize_rangeslice, combinations_step
├── unicode.rs           // utf8 ↔ utf32 helpers (shared)
├── slice.rs             // unchanged
├── prelude.rs           // glob import surface for kernel implementors
│
├── dispatch/
│   ├── mod.rs
│   ├── backend.rs
│   └── device_query.rs
│
├── cpu/                 // see § 2
├── simd/                // mirror of cpu/
├── cuda/                // mirror of cpu/
└── hip/                 // mirror of cpu/
```

**Why move `error/`, `index/`, `utils/`, `unicode.rs` up:** they're
backend-agnostic. The CUDA argsort kernel still needs `ArgsortOrd`; the
SIMD reduce_sum still raises `KernelError`. Today they live under `cpu/`
which forces awkward `use crate::kernels::cpu::error::KernelError` from
non-CPU code.

---

## 2. `cpu/` directory tree

Grouping rule: **one directory per array-type prefix.** Within a directory,
**sub-group by operation family** (`getitem/`, `reduce/`) when ≥ 4 files
share the family — otherwise keep flat.

```
src/kernels/cpu/
├── mod.rs
├── prelude.rs
│
├── bit_masked_array/                       // 2 files → flat
│   ├── mod.rs
│   ├── to_byte_masked_array.rs
│   └── to_indexed_option_array.rs
│
├── byte_masked_array/                      // 8 → flat (reduce subgroup is only 3, not enough)
│   ├── mod.rs
│   ├── getitem_nextcarry.rs
│   ├── getitem_nextcarry_outindex.rs
│   ├── numnull.rs
│   ├── overlay_mask.rs
│   ├── reduce_next.rs                      // _64 dropped — type is in the wrappers
│   ├── reduce_next_nonlocal_nextshifts.rs
│   ├── reduce_next_nonlocal_nextshifts_fromshifts.rs
│   └── to_indexed_option_array.rs
│
├── content/                                // 1 file
│   ├── mod.rs
│   └── getitem_next_missing_jagged_getmaskstartstop.rs
│
├── index/                                  // 3 free-standing index helpers
│   ├── mod.rs
│   ├── nones_as_index.rs
│   ├── rpad_and_clip_axis0.rs
│   └── rpad_and_clip_axis1.rs
│
├── indexed_array/                          // 17 → split: reduce/ subgroup
│   ├── mod.rs
│   ├── fill.rs
│   ├── fill_count.rs
│   ├── flatten_nextcarry.rs
│   ├── flatten_none2empty.rs
│   ├── getitem_nextcarry.rs
│   ├── getitem_nextcarry_outindex.rs
│   ├── index_of_nulls.rs
│   ├── local_preparenext.rs
│   ├── numnull.rs
│   ├── numnull_parents.rs
│   ├── numnull_unique.rs
│   ├── overlay_mask.rs
│   ├── ranges_carry_next.rs
│   ├── ranges_next.rs
│   ├── reduce/
│   │   ├── mod.rs
│   │   ├── next.rs
│   │   ├── next_fix_offsets.rs
│   │   ├── next_nonlocal_nextshifts.rs
│   │   └── next_nonlocal_nextshifts_fromshifts.rs
│   ├── simplify.rs
│   ├── unique_next_index_and_offsets.rs
│   └── validity.rs
│
├── indexed_option_array/                   // 1 file
│   ├── mod.rs
│   └── rpad_and_clip_mask_axis1.rs
│
├── list_array/                             // 22 → split: getitem/ subgroup (13)
│   ├── mod.rs
│   ├── broadcast_tooffsets.rs
│   ├── combinations.rs
│   ├── combinations_length.rs
│   ├── compact_offsets.rs
│   ├── fill.rs
│   ├── getitem/
│   │   ├── mod.rs
│   │   ├── jagged_apply.rs
│   │   ├── jagged_carrylen.rs
│   │   ├── jagged_descend.rs
│   │   ├── jagged_expand.rs
│   │   ├── jagged_numvalid.rs
│   │   ├── jagged_shrink.rs
│   │   ├── next_array.rs
│   │   ├── next_array_advanced.rs
│   │   ├── next_at.rs
│   │   ├── next_range.rs
│   │   ├── next_range_carrylength.rs
│   │   ├── next_range_counts.rs
│   │   └── next_range_spreadadvanced.rs
│   ├── localindex.rs
│   ├── min_range.rs
│   ├── rpad_and_clip_length_axis1.rs
│   ├── rpad_axis1.rs
│   └── validity.rs
│
├── list_offset_array/                      // 14 → split: reduce/ subgroup (7)
│   ├── mod.rs
│   ├── argsort_strings.rs
│   ├── drop_none_indexes.rs
│   ├── flatten_offsets.rs
│   ├── local_preparenext.rs
│   ├── reduce/
│   │   ├── mod.rs
│   │   ├── local_nextparents.rs
│   │   ├── local_outoffsets.rs
│   │   ├── nonlocal_maxcount_offsetscopy.rs
│   │   ├── nonlocal_nextshifts.rs
│   │   ├── nonlocal_nextstarts.rs
│   │   ├── nonlocal_outstartsstops.rs
│   │   └── nonlocal_preparenext.rs
│   ├── rpad_and_clip_axis1.rs
│   ├── rpad_axis1.rs
│   ├── rpad_length_axis1.rs
│   └── to_regular_array.rs
│
├── masked_array/                           // 1 file
│   ├── mod.rs
│   └── getitem_next_jagged_project.rs
│
├── numpy_array/                            // 9 → split: reduce/ subgroup (3), strings/ subgroup (4)
│   ├── mod.rs
│   ├── pad_zero_to_length.rs
│   ├── rearrange_shifted.rs
│   ├── reduce/
│   │   ├── mod.rs
│   │   ├── adjust_starts.rs
│   │   ├── adjust_starts_shifts.rs
│   │   └── mask_byte_masked_array.rs
│   ├── strings/
│   │   ├── mod.rs
│   │   ├── prepare_utf8_to_utf32_padded.rs
│   │   ├── sort_asstrings_uint8.rs
│   │   ├── unique_strings_uint8.rs
│   │   └── utf8_to_utf32_padded.rs
│   ├── subrange_equal.rs                   // generic over T (handles _bool variant)
│   └── (subrange_equal_bool collapsed → typed wrapper inside subrange_equal.rs)
│
├── record_array/                           // 1 file
│   ├── mod.rs
│   └── reduce_nonlocal_outoffsets.rs
│
├── regular_array/                          // 13 → split: getitem/ subgroup (8), reduce/ subgroup (2)
│   ├── mod.rs
│   ├── combinations.rs
│   ├── getitem/
│   │   ├── mod.rs
│   │   ├── carry.rs
│   │   ├── jagged_expand.rs
│   │   ├── next_array.rs
│   │   ├── next_array_advanced.rs
│   │   ├── next_array_regularize.rs
│   │   ├── next_at.rs
│   │   ├── next_range.rs
│   │   └── next_range_spreadadvanced.rs
│   ├── localindex.rs
│   ├── reduce/
│   │   ├── mod.rs
│   │   ├── local_nextparents.rs
│   │   └── nonlocal_preparenext.rs
│   └── rpad_and_clip_axis1.rs
│
├── union_array/                            // 14 → flat
│   ├── mod.rs
│   ├── fillindex.rs
│   ├── fillindex_count.rs
│   ├── fillna.rs
│   ├── filltags.rs
│   ├── filltags_const.rs
│   ├── flatten_combine.rs
│   ├── flatten_length.rs
│   ├── nestedfill_tags_index.rs
│   ├── project.rs
│   ├── regular_index.rs
│   ├── regular_index_getsize.rs
│   ├── simplify.rs
│   ├── simplify_one.rs
│   └── validity.rs
│
├── reduce/                                 // 15+ free-standing reductions
│   ├── mod.rs
│   ├── argmax.rs                           // includes _complex via typed wrapper
│   ├── argmin.rs
│   ├── count.rs
│   ├── countnonzero.rs                     // includes _complex variant
│   ├── max.rs                              // includes _complex variant
│   ├── min.rs                              // includes _complex variant
│   ├── prod.rs                             // includes _bool, _complex variants
│   ├── sum.rs                              // includes _bool, _complex, int_bool variants
│   └── complex_traits.rs                   // shared Complex<T> wrapper
│
├── sort/                                   // 4 files (sort/argsort + ranges)
│   ├── mod.rs
│   ├── argsort.rs                          // ArgsortOrd lives in kernels::traits
│   ├── ranges.rs                           // sorting_ranges
│   ├── ranges_length.rs                    // sorting_ranges_length
│   └── sort.rs
│
├── unique/                                 // 3 files
│   ├── mod.rs
│   ├── offsets.rs
│   ├── ranges.rs                           // generic over T (handles _bool)
│   └── (unique_ranges_bool collapsed → typed wrapper inside ranges.rs)
│
└── misc/                                   // truly free-standing helpers
    ├── mod.rs
    ├── localindex.rs
    └── missing_repeat.rs
```

---

## 3. Naming rules

| C++ name                                      | Rust path                                   | Public fn names |
|-----------------------------------------------|---------------------------------------------|-----------------|
| `awkward_ListArray_getitem_jagged_apply.cpp`  | `list_array/getitem/jagged_apply.rs`        | `list_array_getitem_jagged_apply` (generic core) <br> `list_array32_getitem_jagged_apply_64`, `list_array64_…`, `list_array_u32_…` (typed wrappers — keep the `_64` suffix where the C++ ABI uses it) |
| `awkward_reduce_sum.cpp`                      | `reduce/sum.rs`                             | `reduce_sum` (generic core) + `reduce_sum_int64_int8_64`, `reduce_sum_float64_float64_64`, … |
| `awkward_reduce_sum_bool.cpp`                 | `reduce/sum.rs` (typed wrapper inside)      | `reduce_sum_bool_int32_64`, `reduce_sum_bool_int64_64` |
| `awkward_reduce_sum_complex.cpp`              | `reduce/sum.rs` (under a `mod complex {}`)  | `reduce_sum_complex_float64_float64_64` |
| `awkward_NumpyArray_subrange_equal.cpp` + `_bool` | `numpy_array/subrange_equal.rs`         | `numpy_array_subrange_equal<T>` + typed wrappers including `_bool` |
| `kernel-utils.cpp`                            | `kernels/utils.rs`                          | free helpers |
| `unicode.cpp`                                 | `kernels/unicode.rs`                        | free helpers |

**Drop the `awkward_` prefix everywhere** — it's redundant inside `crate::kernels::*`.

**Drop the `_64` suffix on the generic core** — it indicates the i64 index ABI,
which is captured by the typed wrappers' parameter list, not the generic
function. Keep `_64` on the typed wrappers so callers from the C ABI find
the same names they used in C++.

**Snake_case file paths.** Directory names use snake_case (`list_array`,
not `ListArray`).

---

## 4. Per-file template

Every kernel file has the same shape — it's enforceable and review-friendly.

```rust
// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! One-paragraph description.
//!
//! Corresponds to `src/cpu-kernels/awkward_<original>.cpp`.

use crate::kernels::error::KernelError;        // if it returns Result
use crate::kernels::traits::ArgsortOrd;        // if it needs traits

// ── Generic core ─────────────────────────────────────────────────────────────

/// Doc comment with semantics, parameters, panics, examples (rustdoc).
#[inline]                                      // see § 5
pub fn list_array_compact_offsets<C>(
    tooffsets: &mut [i64],
    fromstarts: &[C],
    fromstops: &[C],
) -> Result<(), KernelError>
where
    C: Copy + Into<i64>,
{
    // …
}

// ── Typed specialisation wrappers (preserve C ABI naming) ────────────────────

macro_rules! impl_typed { … }
impl_typed!(list_array32_compact_offsets_64,   i32);
impl_typed!(list_array_u32_compact_offsets_64, u32);
impl_typed!(list_array64_compact_offsets_64,   i64);

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    // 3-5 small focused tests covering: basic, edge cases, error path.
}
```

The macro for typed wrappers should live in a shared `kernels::utils` (or
`prelude`) so each file doesn't redefine it.

---

## 5. Cross-cutting decisions

**Inlining.** Add `#[inline]` to every generic core. Typed wrappers just
delegate to the core; without `#[inline]` you pay an extra call frame per
kernel invocation and lose monomorphization-driven vectorization.

**Error handling.** All kernels that can fail return `Result<(), KernelError>`.
Kernels that can only fail via assertion (length mismatch) panic — those
preconditions are the caller's responsibility, mirroring C++ behavior. Don't
return `Result<()>` from a kernel that can't actually return `Err` — it muddies
call sites.

**Bounds-check elision.** For numeric-loop kernels (`reduce_*`, `sort`,
`compact_offsets`), prefer iterator forms (`zip`, `enumerate`, slice patterns)
over explicit indexing — they let the compiler skip per-iteration bounds
checks. When indexing is unavoidable, an `assert_eq!` on lengths at the top
of the function is the canonical way to hint to LLVM that the inner
indexing is safe.

**Tests.** In-file `#[cfg(test)]` blocks per kernel for unit tests.
Cross-kernel integration tests and parity checks against the C++ reference
go in `tests/` at the crate root, not inside each kernel module.

**`mod.rs` per directory** re-exports the kernels as a flat namespace
*for that array type*:

```rust
// src/kernels/cpu/list_array/mod.rs
pub mod broadcast_tooffsets;
pub mod combinations;
pub mod combinations_length;
pub mod compact_offsets;
pub mod fill;
pub mod getitem;          // sub-module
pub mod localindex;
pub mod min_range;
pub mod rpad_and_clip_length_axis1;
pub mod rpad_axis1;
pub mod validity;

// Optional flat re-export so callers can write
//   use crate::kernels::cpu::list_array::list_array64_compact_offsets_64;
// instead of
//   use crate::kernels::cpu::list_array::compact_offsets::list_array64_compact_offsets_64;
pub use compact_offsets::*;
pub use combinations::*;
// …
```

The flat `pub use *` is optional but helps callers from Python-binding
code keep imports short.

---

## 6. File-count summary

| Group                 | C++ files | Rust files (proposed) | Notes |
|----------------------|-----------|------------------------|-------|
| bit_masked_array      | 2  | 2  | flat |
| byte_masked_array     | 8  | 8  | flat |
| content               | 1  | 1  | |
| index                 | 3  | 3  | flat |
| indexed_array         | 17 | 16 + reduce/(4) | reduce subgroup |
| indexed_option_array  | 1  | 1  | |
| list_array            | 22 | 9 + getitem/(13) | getitem subgroup |
| list_offset_array     | 14 | 7 + reduce/(7) | reduce subgroup |
| masked_array          | 1  | 1  | |
| numpy_array           | 9  | 2 + reduce/(3) + strings/(4) | _bool variant fused |
| record_array          | 1  | 1  | |
| regular_array         | 13 | 3 + getitem/(8) + reduce/(2) | |
| union_array           | 14 | 14 | flat |
| reduce (free)         | 15+| 9 (variants fused) | _bool, _complex collapsed into typed wrappers |
| sort                  | 4  | 4  | |
| unique                | 3  | 2  | _bool variant fused |
| misc                  | 2  | 2  | |
| **Total**             | **~150** | **~125** | + 17 `mod.rs` files |

Net effect: ~25 fewer files than 1-to-1, mostly by collapsing `_bool` and
`_complex` typed variants into their generic-typed siblings.

---

## 7. Migration plan

You're partway through the restructure on `ianna/cpu_kernels_restructure`.
Order to land it cleanly:

1. **Move `error/`, `index/`, `unicode.rs`, `utils/` up to `src/kernels/`.**
   One mechanical commit. Update import paths via `cargo fix`.

2. **Add `pub mod`s for the new subdirectories in `cpu/mod.rs`.** Right now
   the nested `reduce/`, `numpy_array/`, `misc/`, `union_array/` files exist
   but aren't compiled. Get them building first — even before moving the
   flat files in — to confirm the structure is correct.

3. **Move kernels one array-type-directory at a time.** Each PR/commit moves
   one directory's worth (e.g. all `byte_masked_array_*.rs` → `byte_masked_array/`).
   The mechanical pattern:
   - `git mv src/kernels/cpu/byte_masked_array_foo.rs src/kernels/cpu/byte_masked_array/foo.rs`
   - In `cpu/mod.rs`: replace the flat `pub mod byte_masked_array_foo;` with `pub mod byte_masked_array;`
   - Add `pub mod foo;` to the new `byte_masked_array/mod.rs`
   - Run `cargo test --no-default-features --lib` to confirm.

4. **Collapse `_bool` and `_complex` variants** into their generic siblings
   (last, once everything compiles in the new layout). This is the only
   step that touches kernel internals; everything before is pure file move.

5. **Mirror the same skeleton under `simd/`.** Empty `mod.rs` files at first;
   stubs (`pub fn …() { panic!("simd not yet implemented") }`) so the module
   tree compiles. Fill in actual SIMD as needed.

---

## 8. What's *not* changing

- **The kernels' function signatures.** The Python binding already calls
  `list_array64_compact_offsets_64(…)` etc. The wrapper names stay; only
  their file paths move.
- **The C ABI.** Typed wrappers keep their full names with `_64` etc.,
  so anything that calls the kernels by symbol name still works.
- **Test format.** In-file `#[cfg(test)]` blocks stay in the kernel files.
