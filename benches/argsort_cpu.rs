// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! CPU argsort benchmark: Rust serial vs Rust parallel (rayon) vs C++ serial.
//!
//! Dataset shapes match `argsort_hip_rocprof_all` so timings are directly
//! comparable to the GPU numbers from della-milan.
//!
//! # Run (Rust-only, no GPU required)
//!
//! ```text
//! cargo bench --bench argsort_cpu
//! ```
//!
//! # Run with awkward-cpp side-by-side
//!
//! ```text
//! AWKWARD_CPP_PATH=~/Projects/awkward.2.9.x/awkward/awkward-cpp \
//!     cargo bench --features bench-cxx --bench argsort_cpu
//! ```
//!
//! # Configs
//!
//! Four dataset shapes, each covering a different regime:
//!
//! | label           | nlists  | list lengths | total elems  |
//! |-----------------|---------|--------------|--------------|
//! | small-8-64      | 50 000  | 8 – 64       | ~1.8 M       |
//! | medium-65-256   | 20 000  | 65 – 256     | ~3.2 M       |
//! | large-257-2048  |  5 000  | 257 – 2 048  | ~5.8 M       |
//! | mixed-8-2048    | 20 000  | 8 – 2 048    | ~5.1 M       |
//!
//! The "small-8-64" config directly matches the 50 K small-list workload
//! that was the hardest case for the per-list GPU dispatch loop.

#![allow(clippy::too_many_arguments)]

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use rand::Rng;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Dataset generation
// ---------------------------------------------------------------------------

/// Jagged array of `f32` values with uniform-random list lengths in
/// `[min_len, max_len]`.  Returns `(values, offsets)` where
/// `offsets.len() == nlists + 1` and `offsets[i..i+2]` is the half-open
/// index range of list `i` in `values`.
fn generate_jagged(nlists: usize, min_len: usize, max_len: usize) -> (Vec<f32>, Vec<i64>) {
    let mut rng = rand::thread_rng();
    let mut values = Vec::new();
    let mut offsets = vec![0i64];

    for _ in 0..nlists {
        let len = rng.gen_range(min_len..=max_len);
        for _ in 0..len {
            values.push(rng.r#gen::<f32>());
        }
        offsets.push(values.len() as i64);
    }

    (values, offsets)
}

// ---------------------------------------------------------------------------
// Rust implementations
// ---------------------------------------------------------------------------

/// Single-threaded argsort over a jagged array.
///
/// Initialises `out` with global indices then sorts each segment in-place
/// and makes the indices local (0-based within the list).
fn argsort_serial(values: &[f32], offsets: &[i64], out: &mut [i64]) {
    let nlists = offsets.len() - 1;

    for (i, v) in out.iter_mut().enumerate() {
        *v = i as i64;
    }

    for i in 0..nlists {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        let seg = &mut out[start..end];
        seg.sort_unstable_by(|&a, &b| {
            values[a as usize]
                .partial_cmp(&values[b as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let base = offsets[i];
        for v in seg.iter_mut() {
            *v -= base;
        }
    }
}

/// Rayon parallel argsort — one list per thread-pool task.
///
/// Uses non-overlapping mutable sub-slices of `out` so the sort work is
/// fully parallel with zero extra allocation beyond the rayon thread stack.
///
/// # Safety
///
/// Each worker receives a disjoint `[offsets[i], offsets[i+1])` sub-slice
/// of `out`, so no two threads ever alias the same memory.
fn argsort_parallel(values: &[f32], offsets: &[i64], out: &mut [i64]) {
    let nlists = offsets.len() - 1;

    // Serial initialisation (one cache line per thread would be wasteful).
    for (i, v) in out.iter_mut().enumerate() {
        *v = i as i64;
    }

    // Transmit the raw pointer as a plain integer so it can cross the
    // `Send` boundary.  The safety invariant is maintained because rayon
    // dispatches each list index exactly once (no overlap).
    let out_ptr = out.as_mut_ptr() as usize;

    (0..nlists).into_par_iter().for_each(|i| {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        if start >= end {
            return;
        }

        // SAFETY: segments are non-overlapping by construction of `offsets`.
        let seg = unsafe {
            std::slice::from_raw_parts_mut((out_ptr as *mut i64).add(start), end - start)
        };

        seg.sort_unstable_by(|&a, &b| {
            values[a as usize]
                .partial_cmp(&values[b as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let base = offsets[i];
        for v in seg.iter_mut() {
            *v -= base;
        }
    });
}

// ---------------------------------------------------------------------------
// Optional C++ FFI
// ---------------------------------------------------------------------------

#[cfg(feature = "bench-cxx")]
mod cxx {
    unsafe extern "C" {
        /// Standalone C++ argsort using std::sort (NaN-first, same comparator
        /// as awkward_argsort_float32).  Defined in `benches/argsort_cxx_impl.cpp`,
        /// compiled into `librawkward_argsort_cxx` — no awkward headers needed.
        pub fn bench_awkward_argsort_float32(
            toptr: *mut i64,
            fromptr: *const f32,
            length: i64,
            offsets: *const i64,
            offsetslength: i64,
            ascending: bool,
            stable: bool,
        );
    }
}

// ---------------------------------------------------------------------------
// Benchmark configs
// ---------------------------------------------------------------------------

/// (label, nlists, min_list_len, max_list_len)
///
/// Mirrors the configs in `argsort_hip_rocprof_all.rs` so CPU numbers are
/// directly comparable to the GPU round-trip numbers from della-milan.
const CONFIGS: &[(&str, usize, usize, usize)] = &[
    ("small-8-64", 50_000, 8, 64),
    ("medium-65-256", 20_000, 65, 256),
    ("large-257-2048", 5_000, 257, 2_048),
    ("mixed-8-2048", 20_000, 8, 2_048),
];

// ---------------------------------------------------------------------------
// Criterion benchmark functions
// ---------------------------------------------------------------------------

fn bench_argsort_serial_rust(c: &mut Criterion) {
    let mut group = c.benchmark_group("argsort_serial_rust");

    for &(label, nlists, min_len, max_len) in CONFIGS {
        let (values, offsets) = generate_jagged(nlists, min_len, max_len);
        let total = values.len();
        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(BenchmarkId::new("f32", label), &(), |b, _| {
            let mut out = vec![0i64; total];
            b.iter_batched_ref(
                || (),
                |_| argsort_serial(black_box(&values), black_box(&offsets), black_box(&mut out)),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_argsort_parallel_rust(c: &mut Criterion) {
    let mut group = c.benchmark_group("argsort_parallel_rust");

    for &(label, nlists, min_len, max_len) in CONFIGS {
        let (values, offsets) = generate_jagged(nlists, min_len, max_len);
        let total = values.len();
        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(BenchmarkId::new("f32", label), &(), |b, _| {
            let mut out = vec![0i64; total];
            b.iter_batched_ref(
                || (),
                |_| argsort_parallel(black_box(&values), black_box(&offsets), black_box(&mut out)),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

#[cfg(feature = "bench-cxx")]
fn bench_argsort_serial_cxx(c: &mut Criterion) {
    let mut group = c.benchmark_group("argsort_serial_cxx");

    for &(label, nlists, min_len, max_len) in CONFIGS {
        let (values, offsets) = generate_jagged(nlists, min_len, max_len);
        let total = values.len();
        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(BenchmarkId::new("f32", label), &(), |b, _| {
            let mut out = vec![0i64; total];
            b.iter_batched_ref(
                || (),
                |_| unsafe {
                    cxx::bench_awkward_argsort_float32(
                        black_box(out.as_mut_ptr()),
                        black_box(values.as_ptr()),
                        total as i64,
                        black_box(offsets.as_ptr()),
                        offsets.len() as i64,
                        true,  // ascending
                        false, // unstable — matches argsort_serial's sort_unstable_by
                    );
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

// Provide a no-op stub when bench-cxx is off so criterion_group! compiles.
#[cfg(not(feature = "bench-cxx"))]
fn bench_argsort_serial_cxx(_c: &mut Criterion) {}

// ---------------------------------------------------------------------------
// Wire up criterion
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_argsort_serial_rust,
    bench_argsort_parallel_rust,
    bench_argsort_serial_cxx,
);
criterion_main!(benches);
