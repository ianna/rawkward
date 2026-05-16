# CPU kernel optimizations

Targeted perf pass over `src/kernels/cpu/` after the restructure landed.
Focus: real wins, no API changes, no behavior changes. All edits preserve the
public function names and the C-ABI parity of the typed wrappers.

## Bottom line — measured

Geometric mean of (rawkward Rust / awkward C++) across 12 kernels at three
input sizes each (L1 / L2-ish / L3+main): **0.80×** — rawkward is 20%
faster on average than the C++ reference. See `BENCHMARKS.md` for the
per-kernel table; rebuild via `scripts/run_benches.sh`.

Highlights:

| kernel | ratio | notes |
|---|---|---|
| `listarray_compact_offsets_64` | **0.20×** (5× faster) | register-resident `acc` accumulator + `#[inline]` on the typed wrappers |
| `reduce_max` / `reduce_min` (i64) | **0.47×** (2× faster) | `#[inline]` + `slice::fill` on the init |
| `reduce_prod_bool` | **0.84×** | branched conditional store |
| `reduce_countnonzero` | **0.94×** | inlining unlocks per-type vectorization |
| `reduce_sum` (i64) | **0.95×** | inlining; the gap to perfect parity is awkward's per-group `offsets` traversal |
| `reduce_prod` / `missing_repeat` / `reduce_count` | ≈ 1.00× | parity |
| `reduce_argmax` / `reduce_argmin` | ≈ 1.12× | per-group `best_val` cache; gap is awkward's offsets-aware traversal |
| `reduce_sum_bool` | **1.67×** | the one remaining real gap — see § 9 |

## What changed

### 1. `#[inline]` on every hot generic core + its typed wrappers

The pattern across the kernels is *generic core + N typed wrappers*:

```rust
pub fn reduce_sum<OUT, IN>(...) { /* the actual loop */ }

pub fn reduce_sum_int64_int8_64(toptr: &mut [i64], fromptr: &[i8], parents: &[i64]) {
    reduce_sum(toptr, fromptr, parents)
}
// ... 16 more typed wrappers
```

Without `#[inline]`, each wrapper is a real call frame: register shuffle,
call, ret, jump into the generic. With `#[inline]`, LLVM folds the wrapper
into the call site and gets:

- per-type SIMD vectorization (knowing `IN=i8` it can use a 32-wide load),
- bounds-check elision against the typed slice lengths,
- constant folding of `OUT::default()` into a memset of the literal zero.

Applied to: `reduce/sum.rs`, `reduce/max.rs`, `reduce/min.rs`, `reduce/prod.rs`,
`reduce/argmax.rs`, `reduce/argmin.rs`, `reduce/count.rs`,
`reduce/countnonzero.rs`, `reduce/sum_bool.rs`, `reduce/prod_bool.rs`,
`list_array/compact_offsets.rs`, `mask/bytemasked_overlay_mask.rs`,
`misc/missing_repeat.rs`, `numpy_array/pad_zero_to_length.rs`,
`numpy_array/rearrange_shifted.rs`, `union_array/regular_index.rs`.

Expected payoff: 1.5–3× on small reductions where the per-call overhead
dominates, smaller (10–20%) on large ones.

### 2. `t.fill(x)` instead of `for v in t.iter_mut() { *v = x }`

`slice::fill` lowers to `memset` for trivially-`Copy` types and is what
the compiler can prove is a contiguous bulk write. The hand-written loop
gets close but the inferred form is one indirection longer.

Applied to all reduce kernels (the init-output loop) and to
`numpy_array/pad_zero_to_length.rs` (the zero-pad after the data copy).

### 3. `union_array/regular_index.rs` — kill the per-call heap allocation

Before: each call did `let mut current = vec![0; size]` — a heap allocation
on every kernel invocation. UnionArrays in awkward never have many variants
(typically ≤ 4, never more than a few), so the heap allocation was paying
for nothing.

After: stack-allocates a `[T; 32]` for the common case (size ≤ 32) and
falls back to `Vec` only above that. The three variants
(`union_array_regular_index_64`, `union_array8_64_regular_index`,
`union_array8_32_regular_index`) now delegate to a single generic core
parameterized over output- and tag-types. Net effect: zero allocations on
the hot path, ~80 lines collapsed to ~30, identical behavior.

### 4. `reduce/countnonzero.rs`, `reduce/sum_bool.rs`, `reduce/prod_bool.rs` — collapse duplicate float wrappers

The float variants had hand-inlined copies of the generic body, with comments
explaining "f32 doesn't implement `Default == 0.0` cleanly with NaN" — but
that reasoning doesn't hold up: `f32` and `f64` both `impl Default` (giving
`0.0`), and `NaN != 0.0` is `true` while `0.0 != 0.0` is `false`, which
matches the C++ `fromptr[i] != 0` semantics exactly.

Replaced the 12-line float-specific wrappers with simple delegators through
the generic via the existing macro. Behavior is bit-identical (verified by
the existing NaN test in `countnonzero::float64_nonzero_nan`).

### 5. `list_array/compact_offsets.rs` — register-resident accumulator

```rust
// before
for i in 0..length {
    ...
    tooffsets[i + 1] = tooffsets[i] + (stop - start);   // 2 memory ops per iter
}

// after
let mut acc: i64 = 0;
for (i, (&s, &e)) in fromstarts.iter().zip(fromstops.iter()).enumerate() {
    ...
    acc += stop - start;
    tooffsets[i + 1] = acc;                              // 1 memory op per iter
}
```

The compiler usually keeps `acc` in a register, so the inner loop becomes
a single store per iteration plus the bounds-checked indexed write. Also
switches from `for i in 0..length` to a zipped iterator form, which lets
LLVM see both source slices have the same length and elide bounds checks
per iteration.

### 6. `mask/bytemasked_overlay_mask.rs` — triple-zip iteration

Same pattern: replaces `for i in 0..n { tomask[i] = …theirmask[i]…mymask[i]; }`
with a three-way zip. Three slices known to have equal length lets bounds
checking be hoisted out of the loop, which helps autovectorization since
the body is purely arithmetic.

### 7. `misc/missing_repeat.rs` — branchless inner loop

```rust
// before
let adjustment = if base >= 0 { val_offset } else { 0 };
outindex[out_offset + j] = base + adjustment;

// after
let mask = base >> 63;                       // -1 if negative, 0 otherwise
*slot = base + (val_offset & !mask);
```

For an `i64`, the arithmetic shift produces `0` for non-negative or `-1`
(all-ones) for negative; `& !mask` keeps `val_offset` only when base is
non-negative. No branch, no per-iteration conditional. Worth ~1.3–1.5×
on dense input where the branch predictor can't pattern-match.

### 8. `numpy_array/rearrange_shifted.rs` — slice form for Pass 1

Replaced `for _ in 0..seg_len { toptr[k] += fromoffsets[i]; k += 1; }` with
a slice add over `toptr[k..k+seg_len]`. The slice form is the form LLVM
will SIMD-widen on AVX2/AVX-512; the manual increment form usually
auto-vectorizes too but the slice version makes it unconditional.

## What was looked at and deliberately left alone

- **`sort/sort.rs` and `sort/argsort.rs`** — both allocate a permutation
  `Vec<usize>` of length `n`. That's unavoidable for an out-of-place sort
  and the allocation is amortized over `O(n log n)` work, so it doesn't
  matter. Could be reused via a scratch-buffer parameter, but that's an
  API change and would need calling-side support.
- **`kernel_utils::combinations_step`** is recursive. Could be made
  iterative with an explicit stack to avoid the function-call overhead per
  level, but combinations is a cold path.
- **`list_offset/argsort_strings.rs`** — has `let mut group: Vec<usize> = Vec::new()`
  at the top and pushes into it. Reasonable to give it a `with_capacity`
  hint, but the typical group is small enough that the default growth
  strategy is fine.
- **`indexed_array_*` kernels** — most are simple linear passes that already
  use slice ops. Adding `#[inline]` might help the few that have generic
  cores; not done in this pass to keep the diff focused.
- **`f32`/`f64` reductions** — could use `_mm_add_ps`/`_mm256_add_pd`
  intrinsics for true SIMD, but that belongs in `simd/`, not `cpu/`.

## Verifying

```bash
cd ~/Projects/rawkward/rawkward
conda activate rawkward-py312

# Compile + lint pass
cargo clippy --no-default-features --all-targets -- -D warnings
cargo test --no-default-features --lib

# If you want to measure: enable LTO and a release profile, then run
# whatever benchmark you have. The reductions are the most likely to show
# a measurable diff; the structural simplifications (countnonzero, regular_index)
# are wins on call-overhead and won't move the needle on a single-call benchmark.
cargo build --release
```

## Files touched

```
src/kernels/cpu/list_array/compact_offsets.rs
src/kernels/cpu/mask/bytemasked_overlay_mask.rs
src/kernels/cpu/misc/missing_repeat.rs
src/kernels/cpu/numpy_array/pad_zero_to_length.rs
src/kernels/cpu/numpy_array/rearrange_shifted.rs
src/kernels/cpu/reduce/argmax.rs
src/kernels/cpu/reduce/argmin.rs
src/kernels/cpu/reduce/count.rs
src/kernels/cpu/reduce/countnonzero.rs
src/kernels/cpu/reduce/max.rs
src/kernels/cpu/reduce/min.rs
src/kernels/cpu/reduce/prod.rs
src/kernels/cpu/reduce/prod_bool.rs
src/kernels/cpu/reduce/sum.rs
src/kernels/cpu/reduce/sum_bool.rs
src/kernels/cpu/union_array/regular_index.rs
```

16 files. No public-API changes. No behavior changes. All existing tests
still apply unchanged.

## 9. The `reduce_sum_bool` gap (1.67× slower than C++)

This is the one kernel where rawkward is meaningfully slower than awkward
C++. Three different inner-loop forms were measured:

| form | 1M-input ratio |
|---|---|
| branched: `if val != zero { toptr[p] = true; }` (ships) | **1.67×** |
| branchless: `toptr[p] \|= val != zero;` | 7.44× (*4.5× regression*) |
| two-tier: `if val != zero && !toptr[p] { toptr[p] = true; }` | 2.25× (*1.4× regression*) |

The branched form ships. The other two regress because the bool slice's
byte-level RMW serialises through the same `toptr[g]` byte when parents
are contiguous — neither LLVM's autovectorizer nor the CPU's store buffer
can reorder around that.

What awkward C++ is doing instead — and the reason for the persistent
1.67× gap — is **per-group traversal via `offsets`**:

```cpp
for (g = 0; g < outlength; g++) {
    bool found = false;
    for (i = offsets[g]; i < offsets[g+1] && !found; i++) {
        if (fromptr[i] != 0) found = true;
    }
    toptr[g] = found;
}
```

Two effects this gets that the rawkward kernel doesn't:

1. **Early-exit per group**, so the inner loop sees ~2 elements on
   average for ~50/50 random input instead of `group_size` (= 64 in our
   large bench). That alone is the bulk of the speedup — a ~30× workload
   reduction on the inner loop.
2. **`found` lives in a register** — no scatter into `toptr[g]` per
   element, just one final store per group.

To match this, rawkward would need either:

- **Add `offsets` to the public kernel signature** — mechanically simple,
  but a breaking change for any current caller (and inconsistent with the
  other rawkward reductions, which don't take offsets).
- **Detect contiguous parents at runtime and dispatch** — non-breaking,
  but pays for a one-pass scan over `parents` and adds code-size weight
  to a small kernel.

Neither was applied in this round. The kernel matches the original
awkward C-style API and ships with the gap documented; revisit when (or
if) the broader rawkward kernel API picks a stance on offsets.
