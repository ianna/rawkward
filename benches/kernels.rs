// benches/kernels.rs
//
// Side-by-side benchmarks for the rawkward CPU kernels and their awkward
// C++ counterparts. The C++ side is linked in by build.rs when the
// `bench-cxx` feature is enabled; without that feature, only the Rust
// kernels are exercised.
//
// Run:
//   # Rust-only:
//   cargo bench --no-default-features --bench kernels
//
//   # Side-by-side with the awkward C++ kernels:
//   AWKWARD_CPP_PATH=~/Projects/awkward.2.9.x/awkward/awkward-cpp \
//       cargo bench --no-default-features --features bench-cxx --bench kernels
//
//   # ...then turn the criterion JSON into BENCHMARKS.md:
//   python3 scripts/bench_to_markdown.py
//
// Each kernel is exercised at three input sizes (small/medium/large) so
// the comparison shows where call overhead dominates vs where the inner
// loop wins.

#![allow(clippy::too_many_arguments)]

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};

// ── Inputs ───────────────────────────────────────────────────────────────────

/// Three input sizes. Small fits in L1, medium fits in L2-ish, large spills
/// to L3/main memory. Outlength is set so that on average ~16 elements map
/// to each output group — typical for awkward reductions over jagged data.
const SIZES: &[(usize, usize)] = &[
    (1_024, 128),        // small:  1 K elements, 128 groups   (8 elems/group)
    (65_536, 4_096),     // medium: 64 K elements, 4 K groups  (16 elems/group)
    (1_048_576, 16_384), // large:  1 M elements, 16 K groups (64 elems/group)
];

/// Build a `parents` array where each output group contains a CONTIGUOUS
/// run of input elements. Required by awkward's reduce kernels because
/// they read `offsets[group]` as the start of group `group` in the parents
/// array and walk forward — non-contiguous parents would break that
/// traversal. The Rust kernels don't care either way, so this also serves
/// as their input.
fn make_parents_contig(lenparents: usize, outlength: usize) -> Vec<i64> {
    let group_size = lenparents.div_ceil(outlength).max(1);
    (0..lenparents)
        .map(|i| ((i / group_size).min(outlength - 1)) as i64)
        .collect()
}

/// Cumulative-count offsets matching `make_parents_contig` — length
/// `outlength + 1`, with `offsets[g]` the index in `parents` where group
/// `g` starts and `offsets[outlength] == lenparents`.
fn make_offsets_contig(lenparents: usize, outlength: usize) -> Vec<i64> {
    let group_size = lenparents.div_ceil(outlength).max(1);
    (0..=outlength)
        .map(|g| ((g * group_size).min(lenparents)) as i64)
        .collect()
}

/// `starts[g]` = `offsets[g]` for `g` in `0..outlength`. Awkward's
/// argmax/argmin take this separately even though it's redundant with
/// `offsets`.
fn make_starts_contig(lenparents: usize, outlength: usize) -> Vec<i64> {
    let offsets = make_offsets_contig(lenparents, outlength);
    offsets[..outlength].to_vec()
}

fn make_data_i64(n: usize) -> Vec<i64> {
    // Deterministic LCG so successive runs are reproducible.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        })
        .collect()
}

// ── Optional C++ FFI ─────────────────────────────────────────────────────────
//
// The block below is only compiled when `bench-cxx` is on. The build.rs
// links libawkward_bench_cxx (a static archive of the awkward .cpp
// sources plus our void-returning thunks).

#[cfg(feature = "bench-cxx")]
mod cxx {
    type I64 = i64;
    type Bool = bool;

    unsafe extern "C" {
        pub fn bench_awkward_reduce_sum_int64_int64_64(
            toptr: *mut I64,
            fromptr: *const I64,
            parents: *const I64,
            offsets: *const I64,
            lenparents: I64,
            outlength: I64,
        );
        pub fn bench_awkward_reduce_max_int64_int64_64(
            toptr: *mut I64,
            fromptr: *const I64,
            parents: *const I64,
            offsets: *const I64,
            lenparents: I64,
            outlength: I64,
            identity: I64,
        );
        pub fn bench_awkward_reduce_min_int64_int64_64(
            toptr: *mut I64,
            fromptr: *const I64,
            parents: *const I64,
            offsets: *const I64,
            lenparents: I64,
            outlength: I64,
            identity: I64,
        );
        pub fn bench_awkward_reduce_prod_int64_int64_64(
            toptr: *mut I64,
            fromptr: *const I64,
            parents: *const I64,
            offsets: *const I64,
            lenparents: I64,
            outlength: I64,
        );
        pub fn bench_awkward_reduce_argmax_int64_64(
            toptr: *mut I64,
            fromptr: *const I64,
            parents: *const I64,
            offsets: *const I64,
            lenparents: I64,
            starts: *const I64,
            outlength: I64,
        );
        pub fn bench_awkward_reduce_argmin_int64_64(
            toptr: *mut I64,
            fromptr: *const I64,
            parents: *const I64,
            offsets: *const I64,
            lenparents: I64,
            starts: *const I64,
            outlength: I64,
        );
        pub fn bench_awkward_reduce_count_64(
            toptr: *mut I64,
            parents: *const I64,
            lenparents: I64,
            outlength: I64,
        );
        pub fn bench_awkward_reduce_countnonzero_int64_64(
            toptr: *mut I64,
            fromptr: *const I64,
            parents: *const I64,
            lenparents: I64,
            outlength: I64,
        );
        pub fn bench_awkward_reduce_sum_bool_int64_64(
            toptr: *mut Bool,
            fromptr: *const I64,
            parents: *const I64,
            offsets: *const I64,
            lenparents: I64,
            outlength: I64,
        );
        pub fn bench_awkward_reduce_prod_bool_int64_64(
            toptr: *mut Bool,
            fromptr: *const I64,
            parents: *const I64,
            offsets: *const I64,
            lenparents: I64,
            outlength: I64,
        );
        pub fn bench_awkward_ListArray64_compact_offsets_64(
            tooffsets: *mut I64,
            fromstarts: *const I64,
            fromstops: *const I64,
            length: I64,
        );
        pub fn bench_awkward_missing_repeat_64(
            outindex: *mut I64,
            index: *const I64,
            indexlength: I64,
            repetitions: I64,
            regularsize: I64,
        );
    }
}

// ── Bench helpers ────────────────────────────────────────────────────────────

/// Add a "rust" timer and (when bench-cxx is on) a "cxx" timer to the same
/// criterion BenchmarkGroup so they appear side by side in the output.
macro_rules! bench_pair {
    ($group:expr, $size:expr, $label:expr, $rust:expr, $cxx:expr $(,)?) => {{
        let group = &mut $group;
        group.bench_with_input(BenchmarkId::new("rust", $label), $size, |b, _| {
            b.iter_batched_ref(|| (), |_| $rust, BatchSize::SmallInput)
        });
        #[cfg(feature = "bench-cxx")]
        group.bench_with_input(BenchmarkId::new("cxx", $label), $size, |b, _| {
            b.iter_batched_ref(|| (), |_| $cxx, BatchSize::SmallInput)
        });
        #[cfg(not(feature = "bench-cxx"))]
        let _ = $cxx;
    }};
}

// ── reduce_sum ───────────────────────────────────────────────────────────────

fn bench_reduce_sum(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::sum::reduce_sum_int64_int64_64;

    let mut group = c.benchmark_group("reduce_sum_i64_i64");
    for &(n, k) in SIZES {
        let from = make_data_i64(n);
        let parents = make_parents_contig(n, k);
        let offsets = make_offsets_contig(n, k);
        let mut out_rust = vec![0i64; k];
        let mut out_cxx = vec![0i64; k];
        group.throughput(Throughput::Elements(n as u64));
        bench_pair!(
            group,
            &(n, k),
            format!("{n}/{k}"),
            reduce_sum_int64_int64_64(
                black_box(&mut out_rust),
                black_box(&from),
                black_box(&offsets),
            ),
            {
                #[cfg(feature = "bench-cxx")]
                unsafe {
                    cxx::bench_awkward_reduce_sum_int64_int64_64(
                        black_box(out_cxx.as_mut_ptr()),
                        black_box(from.as_ptr()),
                        black_box(parents.as_ptr()),
                        black_box(offsets.as_ptr()),
                        n as i64,
                        k as i64,
                    );
                }
                let _ = (&mut out_cxx, &parents);
            }
        );
    }
    group.finish();
}

// ── reduce_max / reduce_min (identity arg) ───────────────────────────────────

fn bench_reduce_max(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::max::reduce_max_int64_int64_64;
    let mut group = c.benchmark_group("reduce_max_i64_i64");
    for &(n, k) in SIZES {
        let from = make_data_i64(n);
        let parents = make_parents_contig(n, k);
        let offsets = make_offsets_contig(n, k);
        let mut out_rust = vec![0i64; k];
        let mut out_cxx = vec![0i64; k];
        group.throughput(Throughput::Elements(n as u64));
        bench_pair!(
            group,
            &(n, k),
            format!("{n}/{k}"),
            reduce_max_int64_int64_64(
                black_box(&mut out_rust),
                black_box(&from),
                black_box(&offsets),
                i64::MIN,
            ),
            {
                #[cfg(feature = "bench-cxx")]
                unsafe {
                    cxx::bench_awkward_reduce_max_int64_int64_64(
                        black_box(out_cxx.as_mut_ptr()),
                        black_box(from.as_ptr()),
                        black_box(parents.as_ptr()),
                        black_box(offsets.as_ptr()),
                        n as i64,
                        k as i64,
                        i64::MIN,
                    );
                }
                let _ = (&mut out_cxx, &parents);
            }
        );
    }
    group.finish();
}

fn bench_reduce_min(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::min::reduce_min_int64_int64_64;
    let mut group = c.benchmark_group("reduce_min_i64_i64");
    for &(n, k) in SIZES {
        let from = make_data_i64(n);
        let parents = make_parents_contig(n, k);
        let offsets = make_offsets_contig(n, k);
        let mut out_rust = vec![0i64; k];
        let mut out_cxx = vec![0i64; k];
        group.throughput(Throughput::Elements(n as u64));
        bench_pair!(
            group,
            &(n, k),
            format!("{n}/{k}"),
            reduce_min_int64_int64_64(
                black_box(&mut out_rust),
                black_box(&from),
                black_box(&offsets),
                i64::MAX,
            ),
            {
                #[cfg(feature = "bench-cxx")]
                unsafe {
                    cxx::bench_awkward_reduce_min_int64_int64_64(
                        black_box(out_cxx.as_mut_ptr()),
                        black_box(from.as_ptr()),
                        black_box(parents.as_ptr()),
                        black_box(offsets.as_ptr()),
                        n as i64,
                        k as i64,
                        i64::MAX,
                    );
                }
                let _ = (&mut out_cxx, &parents);
            }
        );
    }
    group.finish();
}

// ── reduce_prod ──────────────────────────────────────────────────────────────

fn bench_reduce_prod(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::prod::reduce_prod_int64_int64_64;
    let mut group = c.benchmark_group("reduce_prod_i64_i64");
    for &(n, k) in SIZES {
        let from = make_data_i64(n);
        let parents = make_parents_contig(n, k);
        let offsets = make_offsets_contig(n, k);
        let mut out_rust = vec![0i64; k];
        let mut out_cxx = vec![0i64; k];
        group.throughput(Throughput::Elements(n as u64));
        bench_pair!(
            group,
            &(n, k),
            format!("{n}/{k}"),
            reduce_prod_int64_int64_64(
                black_box(&mut out_rust),
                black_box(&from),
                black_box(&offsets),
            ),
            {
                #[cfg(feature = "bench-cxx")]
                unsafe {
                    cxx::bench_awkward_reduce_prod_int64_int64_64(
                        black_box(out_cxx.as_mut_ptr()),
                        black_box(from.as_ptr()),
                        black_box(parents.as_ptr()),
                        black_box(offsets.as_ptr()),
                        n as i64,
                        k as i64,
                    );
                }
                let _ = (&mut out_cxx, &parents);
            }
        );
    }
    group.finish();
}

// ── reduce_argmax / reduce_argmin (have BOTH offsets and starts) ─────────────

fn bench_reduce_argmax(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::argmax::reduce_argmax_int64_64;
    let mut group = c.benchmark_group("reduce_argmax_i64");
    for &(n, k) in SIZES {
        let from = make_data_i64(n);
        let parents = make_parents_contig(n, k);
        let offsets = make_offsets_contig(n, k);
        let starts = make_starts_contig(n, k);
        let mut out_rust = vec![0i64; k];
        let mut out_cxx = vec![0i64; k];
        group.throughput(Throughput::Elements(n as u64));
        bench_pair!(
            group,
            &(n, k),
            format!("{n}/{k}"),
            reduce_argmax_int64_64(
                black_box(&mut out_rust),
                black_box(&from),
                black_box(&offsets),
            ),
            {
                #[cfg(feature = "bench-cxx")]
                unsafe {
                    cxx::bench_awkward_reduce_argmax_int64_64(
                        black_box(out_cxx.as_mut_ptr()),
                        black_box(from.as_ptr()),
                        black_box(parents.as_ptr()),
                        black_box(offsets.as_ptr()),
                        n as i64,
                        black_box(starts.as_ptr()),
                        k as i64,
                    );
                }
                let _ = (&mut out_cxx, &parents, &starts);
            }
        );
    }
    group.finish();
}

fn bench_reduce_argmin(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::argmin::reduce_argmin_int64_64;
    let mut group = c.benchmark_group("reduce_argmin_i64");
    for &(n, k) in SIZES {
        let from = make_data_i64(n);
        let parents = make_parents_contig(n, k);
        let offsets = make_offsets_contig(n, k);
        let starts = make_starts_contig(n, k);
        let mut out_rust = vec![0i64; k];
        let mut out_cxx = vec![0i64; k];
        group.throughput(Throughput::Elements(n as u64));
        bench_pair!(
            group,
            &(n, k),
            format!("{n}/{k}"),
            reduce_argmin_int64_64(
                black_box(&mut out_rust),
                black_box(&from),
                black_box(&offsets),
            ),
            {
                #[cfg(feature = "bench-cxx")]
                unsafe {
                    cxx::bench_awkward_reduce_argmin_int64_64(
                        black_box(out_cxx.as_mut_ptr()),
                        black_box(from.as_ptr()),
                        black_box(parents.as_ptr()),
                        black_box(offsets.as_ptr()),
                        n as i64,
                        black_box(starts.as_ptr()),
                        k as i64,
                    );
                }
                let _ = (&mut out_cxx, &parents, &starts);
            }
        );
    }
    group.finish();
}

// ── reduce_count (no offsets) ────────────────────────────────────────────────

fn bench_reduce_count(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::count::reduce_count_64;
    let mut group = c.benchmark_group("reduce_count_64");
    for &(n, k) in SIZES {
        let parents = make_parents_contig(n, k);
        let mut out_rust = vec![0i64; k];
        let mut out_cxx = vec![0i64; k];
        group.throughput(Throughput::Elements(n as u64));
        bench_pair!(
            group,
            &(n, k),
            format!("{n}/{k}"),
            reduce_count_64(black_box(&mut out_rust), black_box(&parents)),
            {
                #[cfg(feature = "bench-cxx")]
                unsafe {
                    cxx::bench_awkward_reduce_count_64(
                        black_box(out_cxx.as_mut_ptr()),
                        black_box(parents.as_ptr()),
                        n as i64,
                        k as i64,
                    );
                }
                let _ = &mut out_cxx;
            }
        );
    }
    group.finish();
}

// ── reduce_countnonzero (no offsets) ─────────────────────────────────────────

fn bench_reduce_countnonzero(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::countnonzero::reduce_countnonzero_int64_64;
    let mut group = c.benchmark_group("reduce_countnonzero_i64");
    for &(n, k) in SIZES {
        let from = make_data_i64(n);
        let parents = make_parents_contig(n, k);
        let mut out_rust = vec![0i64; k];
        let mut out_cxx = vec![0i64; k];
        group.throughput(Throughput::Elements(n as u64));
        bench_pair!(
            group,
            &(n, k),
            format!("{n}/{k}"),
            reduce_countnonzero_int64_64(
                black_box(&mut out_rust),
                black_box(&from),
                black_box(&parents),
            ),
            {
                #[cfg(feature = "bench-cxx")]
                unsafe {
                    cxx::bench_awkward_reduce_countnonzero_int64_64(
                        black_box(out_cxx.as_mut_ptr()),
                        black_box(from.as_ptr()),
                        black_box(parents.as_ptr()),
                        n as i64,
                        k as i64,
                    );
                }
                let _ = &mut out_cxx;
            }
        );
    }
    group.finish();
}

// ── reduce_sum_bool / reduce_prod_bool (have offsets) ────────────────────────

fn bench_reduce_sum_bool(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::sum_bool::reduce_sum_bool_int64_64;
    let mut group = c.benchmark_group("reduce_sum_bool_i64");
    for &(n, k) in SIZES {
        let from = make_data_i64(n);
        let parents = make_parents_contig(n, k);
        let offsets = make_offsets_contig(n, k);
        let mut out_rust = vec![false; k];
        let mut out_cxx = vec![false; k];
        group.throughput(Throughput::Elements(n as u64));
        bench_pair!(
            group,
            &(n, k),
            format!("{n}/{k}"),
            reduce_sum_bool_int64_64(
                black_box(&mut out_rust),
                black_box(&from),
                black_box(&offsets),
            ),
            {
                #[cfg(feature = "bench-cxx")]
                unsafe {
                    cxx::bench_awkward_reduce_sum_bool_int64_64(
                        black_box(out_cxx.as_mut_ptr()),
                        black_box(from.as_ptr()),
                        black_box(parents.as_ptr()),
                        black_box(offsets.as_ptr()),
                        n as i64,
                        k as i64,
                    );
                }
                let _ = (&mut out_cxx, &parents);
            }
        );
    }
    group.finish();
}

fn bench_reduce_prod_bool(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::prod_bool::reduce_prod_bool_int64_64;
    let mut group = c.benchmark_group("reduce_prod_bool_i64");
    for &(n, k) in SIZES {
        let from = make_data_i64(n);
        let parents = make_parents_contig(n, k);
        let offsets = make_offsets_contig(n, k);
        let mut out_rust = vec![false; k];
        let mut out_cxx = vec![false; k];
        group.throughput(Throughput::Elements(n as u64));
        bench_pair!(
            group,
            &(n, k),
            format!("{n}/{k}"),
            reduce_prod_bool_int64_64(
                black_box(&mut out_rust),
                black_box(&from),
                black_box(&offsets),
            ),
            {
                #[cfg(feature = "bench-cxx")]
                unsafe {
                    cxx::bench_awkward_reduce_prod_bool_int64_64(
                        black_box(out_cxx.as_mut_ptr()),
                        black_box(from.as_ptr()),
                        black_box(parents.as_ptr()),
                        black_box(offsets.as_ptr()),
                        n as i64,
                        k as i64,
                    );
                }
                let _ = (&mut out_cxx, &parents);
            }
        );
    }
    group.finish();
}

// ── ListArray::compact_offsets ───────────────────────────────────────────────

fn bench_compact_offsets(c: &mut Criterion) {
    use rawkward::kernels::cpu::list_array::compact_offsets::list_array64_compact_offsets_64;
    let mut group = c.benchmark_group("listarray_compact_offsets_64");
    for &(n, _) in SIZES {
        // Synthetic monotonically-increasing starts/stops representing
        // a jagged array with average sublist length 8.
        let starts: Vec<i64> = (0..n).map(|i| (i as i64) * 8).collect();
        let stops: Vec<i64> = (0..n).map(|i| (i as i64) * 8 + 8).collect();
        let mut out_rust = vec![0i64; n + 1];
        let mut out_cxx = vec![0i64; n + 1];
        group.throughput(Throughput::Elements(n as u64));
        bench_pair!(
            group,
            &(n, n),
            format!("{n}"),
            list_array64_compact_offsets_64(
                black_box(&mut out_rust),
                black_box(&starts),
                black_box(&stops),
            )
            .unwrap(),
            {
                #[cfg(feature = "bench-cxx")]
                unsafe {
                    cxx::bench_awkward_ListArray64_compact_offsets_64(
                        black_box(out_cxx.as_mut_ptr()),
                        black_box(starts.as_ptr()),
                        black_box(stops.as_ptr()),
                        n as i64,
                    );
                }
                let _ = &mut out_cxx;
            }
        );
    }
    group.finish();
}

// ── missing_repeat ───────────────────────────────────────────────────────────

fn bench_missing_repeat(c: &mut Criterion) {
    use rawkward::kernels::cpu::misc::missing_repeat::missing_repeat_64;
    let mut group = c.benchmark_group("missing_repeat_64");
    for &(n, _) in SIZES {
        let indexlength = (n / 32).max(8);
        let repetitions = 32_i64;
        let regularsize = 16_i64;
        // Mix non-negative indices and -1 nulls roughly 50/50.
        let index: Vec<i64> = (0..indexlength)
            .map(|i| if i % 2 == 0 { i as i64 } else { -1 })
            .collect();
        let outlen = (indexlength as i64 * repetitions) as usize;
        let mut out_rust = vec![0i64; outlen];
        let mut out_cxx = vec![0i64; outlen];
        group.throughput(Throughput::Elements(outlen as u64));
        bench_pair!(
            group,
            &(n, n),
            format!("{indexlength}x{repetitions}"),
            missing_repeat_64(
                black_box(&mut out_rust),
                black_box(&index),
                repetitions,
                regularsize,
            ),
            {
                #[cfg(feature = "bench-cxx")]
                unsafe {
                    cxx::bench_awkward_missing_repeat_64(
                        black_box(out_cxx.as_mut_ptr()),
                        black_box(index.as_ptr()),
                        indexlength as i64,
                        repetitions,
                        regularsize,
                    );
                }
                let _ = &mut out_cxx;
            }
        );
    }
    group.finish();
}

// ── Wire up criterion ────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_reduce_sum,
    bench_reduce_max,
    bench_reduce_min,
    bench_reduce_prod,
    bench_reduce_argmax,
    bench_reduce_argmin,
    bench_reduce_count,
    bench_reduce_countnonzero,
    bench_reduce_sum_bool,
    bench_reduce_prod_bool,
    bench_compact_offsets,
    bench_missing_repeat,
);
criterion_main!(benches);
