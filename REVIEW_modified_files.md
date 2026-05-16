# Review of modified files on `ianna/cpu_kernels_restructure`

Scope: 53 modified files + 3 deleted, vs `git HEAD`. Static review only — this
sandbox has no Rust toolchain. See **§ Verify locally** for the exact commands
to run inside `rawkward-py312`.

Findings are grouped by severity. Items marked **[Bug]** are likely
behaviour-affecting; **[Build]** breaks or significantly changes the build
graph; **[Smell]** is style/maintainability; **[Perf]** is runtime/memory.

---

## 1. Critical — restructured kernels are not wired into the build  [Build]

The branch name is `cpu_kernels_restructure` and you have substantial new code
under nested directories:

```
src/kernels/cpu/reduce/         (13 files: prod.rs, sum.rs, max.rs, min.rs, …)
src/kernels/cpu/numpy_array/    (10 files: pad_zero_to_length.rs, …)
src/kernels/cpu/misc/           ( 5 files: missing_repeat.rs, …)
src/kernels/cpu/union_array/    ( 9 files: regular_index.rs, …)
src/kernels/cpu/list_array/, list_offset/, regular_array/, mask/, index/
```

But `src/kernels/cpu/mod.rs` only declares the **flat** modules
(`pub mod reduce_prod;`, `pub mod numpy_array_pad_zero_to_length;`, …). It
contains no `pub mod reduce;`, `pub mod numpy_array;`, `pub mod misc;`,
`pub mod union_array;` declarations, and the nested directories don't have
their own `mod.rs`.

**Effect:** every edit you made under `reduce/`, `numpy_array/`, `misc/`,
`union_array/`, etc. is dead code. `cargo build` doesn't see those files,
`cargo test` doesn't run their tests, and `cargo clippy` doesn't lint them.
The actual implementations being compiled are still the flat ones at the top
of `cpu/`.

The 41 substantial implementations (e.g. `reduce/prod.rs`, `union_array/regular_index.rs`,
`misc/missing_repeat.rs`, `numpy_array/pad_zero_to_length.rs`,
`numpy_array/rearrange_shifted.rs`) all look correct and well-tested in
isolation, but they aren't built.

**Fix:** decide on the layout first, then either

- **Adopt the nested layout:** add `mod.rs` files under each subdirectory
  re-exporting the kernel functions, replace the flat declarations in
  `cpu/mod.rs` with `pub mod reduce; pub mod numpy_array; …`, and delete the
  old flat files. This is the cleanest end-state.
- **Keep flat for now:** delete the new nested files and continue editing the
  flat ones. Less disruptive if the restructure isn't ready to land.

Whichever you pick, don't ship both — duplicate symbol implementations are a
maintenance trap even when they're not both compiled.

---

## 2. Cargo.toml regressions  [Build]

```
- default = ["python"]
+ default = []

- pyo3 = { ..., features = ["abi3-py39"] }
+ pyo3 = { ..., features = ["extension-module", "auto-initialize", "abi3-py310"] }

- bin_testing = ["pyo3/auto-initialize"]
- (feature removed)

- reqwest = { ..., optional = true }
- tokio   = { ..., optional = true }
+ reqwest = { ... }                      # always-on
+ tokio   = { ... }                      # always-on

- zstd = { version = "0.13", features = ["zstdmt"] }
+ zstd = "0.13"

- [dev-dependencies] proptest, criterion
- [[example]] basic_usage
- (all removed)
```

Concrete issues, in order of severity:

1. **`reqwest` and `tokio` lost their `optional = true` gating.** They were
   gated behind a `remote` feature; now they compile unconditionally. Neither
   is used anywhere in `src/` (I greped — zero `use reqwest`, zero `use tokio`).
   That's roughly 100+ transitive crates and ~30 s of cold compile time you're
   paying for nothing. Restore the optional gating, or — since they're unused —
   drop them entirely until the remote feature actually exists.

2. **`extension-module` + `auto-initialize` together is a misconfiguration.**
   `extension-module` is for *being loaded into* a Python interpreter;
   `auto-initialize` is for *embedding* a Python interpreter. They serve
   opposite roles. The old setup correctly gated `auto-initialize` behind a
   separate `bin_testing` feature so it only turned on for binary tests. The
   new always-on combo can cause link errors or symbol clashes on macOS.

3. **`default = []` is a workflow change.** Previously `cargo build` produced
   the Python module. Now you must say `cargo build --features python` or use
   `maturin develop`. Fine if intentional, but the old example required
   `["python"]` (`required-features = ["python"]`) and that gate is gone too.

4. **`abi3-py39` → `abi3-py310`** raises the minimum Python from 3.9 to 3.10.
   Aligns with `rawkward-py312`, but downstream users on 3.9 will break.
   Document it in the changelog.

5. **`zstdmt` dropped.** Multithreaded zstd compression is gone. Doesn't
   matter today (zstd isn't used), but if/when you wire compression in, you'll
   want it back.

6. **`proptest` and `criterion` removed.** Property tests and benchmarks
   can no longer be written. If that was a deliberate cleanup, fine; if not,
   restore them — both are dev-only and don't affect the published artifact.

7. **`required-features = ["python"]` example removed.** If
   `examples/basic_usage.rs` still exists and uses pyo3 types, `cargo build
   --examples` will now fail when `--features python` isn't passed.

8. **Missing trailing newline** at EOF — POSIX-non-compliant; many tools warn.

**Other unused dependencies in this Cargo.toml:** `arrow`, `arrow-buffer`,
`rayon`, `flate2`, `lz4_flex`, `xxhash-rust`, `xz2`, `zstd`. None appear in
any `use` statement under `src/`. Together they pull in dozens of transitive
crates. Either start using them or remove them.

---

## 3. Behaviour change in `index_rpad_and_clip_axis0.rs`  [Bug]

```rust
// before
pub fn index_rpad_and_clip_axis0(toindex: &mut [i64], target: usize, length: usize) {
    let shorter = target.min(length);
    for i in 0..shorter        { toindex[i] = i as i64; }
    for i in shorter..target   { toindex[i] = -1; }
}

// after
pub fn index_rpad_and_clip_axis0(toindex: &mut [i64], target: usize, length: usize) {
    let shorter = target.min(length);
    let values = (0..shorter as i64).chain(std::iter::repeat(-1));
    toindex.iter_mut().zip(values).for_each(|(slot, val)| { *slot = val; });
}
```

The old version writes exactly `target` slots and never touches `toindex[target..]`.
The new version walks the full `toindex.len()` via `iter_mut()`, so if
`toindex.len() > target` the trailing slots get clobbered with `-1`. The
`target` parameter is now effectively ignored for the upper bound.

If callers always size `toindex` to exactly `target`, the change is invisible.
If any caller passes a larger buffer (e.g. reusing a scratch slice), this is
a silent overwrite of data past the intended end.

**Fix:** restrict the iteration to the first `target` slots:

```rust
pub fn index_rpad_and_clip_axis0(toindex: &mut [i64], target: usize, length: usize) {
    let shorter = target.min(length);
    let dst = &mut toindex[..target];
    let values = (0..shorter as i64).chain(std::iter::repeat(-1));
    dst.iter_mut().zip(values).for_each(|(slot, val)| *slot = val);
}
```

Also worth keeping the original two-loop form — it's simpler, easier for the
optimizer to vectorize, and avoids the cost of the `chain(repeat)` adapter.
The "iterator-fancy" version is not faster here.

---

## 4. Sweep rewrite: `assert!(x.len() >= n + 1)` → `assert!(x.len() > n)`  [Smell]

Affects ~15 files (`list_array_compact_offsets.rs`, `list_array_getitem_jagged_apply.rs`,
`list_array_getitem_jagged_descend.rs`, `list_array_combinations_length.rs`,
`indexed_array_flatten_none2empty.rs`, `union_array_flatten_combine.rs`,
`list_offset_array_reduce_local_outoffsets_64.rs`, …).

For non-negative integers `len >= n + 1` and `len > n` are mathematically
equivalent and compile to the same machine code. The original form is more
self-documenting in this context — these kernels iterate `0..n` and write to
both `tooffsets[i]` and `tooffsets[i + 1]`, so "needs at least `n + 1` slots"
is the natural reading. The new form obscures that.

Not a bug, but a readability regression. If you're going to mass-edit these,
the better direction is the opposite: write
`assert!(tooffsets.len() >= n + 1, "needs n+1 slots for the sentinel")`.

---

## 5. Stray-`/` syntax error fixes  [Build]

Good news: 15 files that started with a stray `/` on line 1 (a hard syntax
error) are now fixed in this branch. These previously prevented the crate
from compiling at all. Affected files include `argsort.rs`, `sort.rs`,
`sorting_ranges.rs`, `unique_offsets.rs`, `numpy_array_subrange_equal.rs`,
`reduce_complex.rs`, `record_array_reduce_nonlocal_outoffsets_64.rs`, etc.
No action needed — just calling out the win.

---

## 6. Doc-comment indentation fixes  [Smell]

Several files (`reduce_prod.rs`, `union_array_validity.rs`, …) re-flow doc
comments to align continuation lines with three spaces instead of the
hanging eleven-space indent. Looks like a `rustfmt`/`cargo fmt` pass.
Pure cosmetic, no concern.

---

## 7. New kernel implementations — quick spot review

The substantive new code under nested dirs (which, again, isn't compiled
yet — see § 1) looks solid. Notes per file:

**`reduce/prod.rs`** — Defines a private `One` trait to avoid `num-traits`.
Reasonable choice. Consider `#[inline]` on `reduce_prod` (the generic core
the typed wrappers delegate to) so it inlines through the wrapper.

**`union_array/regular_index.rs`** — Allocates a `Vec<i64>` of size `size`
per call. For the common case where `size` is small (≤ 4) you could keep this
on the stack with `smallvec` or a fixed-size array. Not worth optimizing
unless this is in a hot path.

**`misc/missing_repeat.rs`** — Inner loop has a per-iteration `if base >= 0`
branch. For a tight kernel like this, the branch predictor will handle it
well, but the branchless form
```rust
let mask = (base >> 63) as i64;          // -1 if negative, 0 otherwise
outindex[out_offset + j] = base + (val_offset & !mask);
```
shaves a branch per element and tends to vectorize more cleanly.

**`numpy_array/pad_zero_to_length.rs`** — Uses `copy_from_slice` and a manual
zero-fill loop. The zero-fill loop could be `slice::fill(T::default())` for
parity (compiles to memset for trivially-copyable types).

**`numpy_array/rearrange_shifted.rs`** — Pass 1 uses an inner `for _ in 0..seg_len { toptr[k] += offset; k += 1; }`
which is correct but inelegant. Slice form is faster and clearer:
```rust
let dst = &mut toptr[k..k + seg_len];
for v in dst.iter_mut() { *v += fromoffsets[i]; }
k += seg_len;
```
The compiler should auto-vectorize that into AVX adds for `i64`.

**`misc/index_rpad_and_clip_axis0.rs`** — Same `iter_mut().zip(chain(repeat))`
pattern as § 3, with the same caveat: it walks the entire output slice rather
than just `target`. If/when you wire this in, fix it the same way.

---

## 8. Inlining hint missing on hot generics  [Perf]

Across the kernels module (≈330 files) only 4 have any `#[inline]` attribute.
Generic kernels like `reduce_sum`, `reduce_max`, `reduce_prod` are called
through monomorphic typed wrappers (`reduce_sum_int64_int8_64`, etc.) which
themselves are called from external code. Without `#[inline]` on the generic
core, the wrapper is a real call frame: arguments shuffled into registers,
ret, then a second jump into the actual loop.

Adding `#[inline]` (or `#[inline(always)]` on the very hot ones) lets LLVM
fold the wrapper into the call site and unlock per-call-site specializations
(loop unrolling, vectorization with known type widths). Expected payoff is
single-digit percent on most kernels but can be 2× on the smallest ones.

Targeted suggestion: add `#[inline]` to every generic `pub fn` in
`reduce_*.rs`, `argsort.rs`, `sort.rs`, `list_array_compact_offsets.rs`, and
the `_64` typed wrappers.

---

## 9. Verify locally

In your `rawkward-py312` conda env, on the host:

```bash
cd ~/Projects/rawkward/rawkward
conda activate rawkward-py312

# 1. Does the crate even compile right now (post-stray-slash fixes)?
cargo check --no-default-features

# 2. With the python feature on (matches what maturin uses):
cargo check --features python

# 3. Lint pass — flags many of the patterns above automatically.
cargo clippy --all-targets --all-features -- -D warnings

# 4. Run the kernel tests (the in-file #[cfg(test)] blocks).
cargo test --no-default-features --lib

# 5. Build the Python module and re-run any python-side tests.
maturin develop --release
pytest python/

# 6. (Optional) Confirm § 1 — the nested files don't show up in the build:
cargo rustc --lib -- --emit=metadata 2>&1 | grep -E "(reduce|numpy_array|misc|union_array)/" || \
    echo "Nested kernels are NOT being compiled (matches §1)."
```

Step 1 will tell you whether anything else is broken besides the stray `/`s.
Step 3 (`cargo clippy`) will surface the `assert!(x.len() > n)` pattern, the
unused `tokio`/`reqwest`/`arrow`/etc. dependencies, and the missing
`#[inline]` hints from § 8.

---

## Summary — recommended order of action

1. **Decide the layout (§ 1).** This blocks everything else; no point
   reviewing the new code if it isn't going to be compiled.
2. **Fix `Cargo.toml` (§ 2).** Re-gate `tokio`/`reqwest`, drop the
   `extension-module + auto-initialize` combo, restore `dev-dependencies` and
   the example's `required-features`, decide on `default` features.
3. **Patch `index_rpad_and_clip_axis0.rs` (§ 3).** Real semantic regression.
4. **Run `cargo clippy` (§ 9).** Will surface the rest of § 4, § 8 and any
   genuinely-new lints.
5. **Style/perf cleanup (§ 4, 7, 8).** Low priority but cheap.
