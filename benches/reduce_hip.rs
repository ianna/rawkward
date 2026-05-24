// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Criterion benchmarks: CPU Rust scalar vs HIP GPU for all segmented
//! reduce kernels (argmax, argmin, count, countnonzero, max, min, prod, sum).
//!
//! Run on della-milan:
//!
//! ```text
//! cargo bench --features hip --bench reduce_hip
//! ```
//!
//! Criterion generates per-group HTML reports under
//! `target/criterion/<group_name>/report/index.html`.
//!
//! # What is measured
//!
//! `cpu`  — pure in-process scalar reduction on already-resident CPU data.
//! `hip`  — device buffers pre-uploaded once; per-iteration cost is kernel
//!          dispatch + synchronous D2H download (no H2D).  Models steady-state
//!          throughput when device buffers are kept alive across calls.
//!
//! # Input sizes
//!
//! ```text
//!   small   1 K elems,  128 segments  (avg  8 elems/seg)
//!   medium 64 K elems, 4 K segments   (avg 16 elems/seg)
//!   large   1 M elems, 16 K segments  (avg 64 elems/seg)
//! ```

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};

#[cfg(all(feature = "hip", hip_rocm))]
use rawkward::backend::{GpuBackend, hip::HipBackend};

// ── Input sizes ───────────────────────────────────────────────────────────────

const SIZES: &[(usize, usize)] = &[(1_024, 128), (65_536, 4_096), (1_048_576, 16_384)];

// ── Data generation ───────────────────────────────────────────────────────────

/// Contiguous-group offsets: `offsets[g]..offsets[g+1]` is group `g`.
fn make_offsets_contig(lenparents: usize, outlength: usize) -> Vec<i64> {
    let group_size = lenparents.div_ceil(outlength).max(1);
    (0..=outlength)
        .map(|g| ((g * group_size).min(lenparents)) as i64)
        .collect()
}

/// Contiguous-group parents: element `i` belongs to group `i / group_size`.
/// Required by `reduce_count_64` and `reduce_countnonzero`.
fn make_parents_contig(lenparents: usize, outlength: usize) -> Vec<i64> {
    let group_size = lenparents.div_ceil(outlength).max(1);
    (0..lenparents)
        .map(|i| (i / group_size).min(outlength - 1) as i64)
        .collect()
}

/// Deterministic i64 data via LCG (same seed as reduce_sum_hip.rs).
fn make_data_i64(n: usize) -> Vec<i64> {
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

// ── reduce_argmax ─────────────────────────────────────────────────────────────

fn bench_reduce_argmax(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::argmax::reduce_argmax;

    #[cfg(all(feature = "hip", hip_rocm))]
    use rawkward::kernels::hip::reduce::argmax::segmented_argmax_i64;

    #[cfg(all(feature = "hip", hip_rocm))]
    let backend = HipBackend::new();

    let mut group = c.benchmark_group("reduce_argmax_hip_i64");

    for &(n, k) in SIZES {
        let data = make_data_i64(n);
        let offsets = make_offsets_contig(n, k);
        let label = format!("{n}/{k}");
        group.throughput(Throughput::Elements(n as u64));

        // ── cpu ──────────────────────────────────────────────────────────────
        let mut out_cpu = vec![0i64; k];
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter_batched_ref(
                || (),
                |_| {
                    reduce_argmax::<i64>(
                        black_box(&mut out_cpu),
                        black_box(&data),
                        black_box(&offsets),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        // ── hip ──────────────────────────────────────────────────────────────
        #[cfg(all(feature = "hip", hip_rocm))]
        {
            let dev_data = backend.upload_slice::<i64>(&data);
            let dev_offsets = backend.upload_slice::<i64>(&offsets);
            let mut dev_out = unsafe { backend.alloc_slice::<i64>(k) };
            let mut out_host = vec![0i64; k];

            group.bench_with_input(BenchmarkId::new("hip", &label), &(), |b, _| {
                b.iter_batched_ref(
                    || (),
                    |_| {
                        segmented_argmax_i64(
                            black_box(&backend),
                            black_box(&dev_data),
                            black_box(&dev_offsets),
                            black_box(&mut dev_out),
                            k as i64,
                        )
                        .expect("HIP reduce_argmax_i64 failed");
                        backend.download_slice(&dev_out, black_box(&mut out_host));
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

// ── reduce_argmin ─────────────────────────────────────────────────────────────

fn bench_reduce_argmin(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::argmin::reduce_argmin;

    #[cfg(all(feature = "hip", hip_rocm))]
    use rawkward::kernels::hip::reduce::argmin::segmented_argmin_i64;

    #[cfg(all(feature = "hip", hip_rocm))]
    let backend = HipBackend::new();

    let mut group = c.benchmark_group("reduce_argmin_hip_i64");

    for &(n, k) in SIZES {
        let data = make_data_i64(n);
        let offsets = make_offsets_contig(n, k);
        let label = format!("{n}/{k}");
        group.throughput(Throughput::Elements(n as u64));

        let mut out_cpu = vec![0i64; k];
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter_batched_ref(
                || (),
                |_| {
                    reduce_argmin::<i64>(
                        black_box(&mut out_cpu),
                        black_box(&data),
                        black_box(&offsets),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        #[cfg(all(feature = "hip", hip_rocm))]
        {
            let dev_data = backend.upload_slice::<i64>(&data);
            let dev_offsets = backend.upload_slice::<i64>(&offsets);
            let mut dev_out = unsafe { backend.alloc_slice::<i64>(k) };
            let mut out_host = vec![0i64; k];

            group.bench_with_input(BenchmarkId::new("hip", &label), &(), |b, _| {
                b.iter_batched_ref(
                    || (),
                    |_| {
                        segmented_argmin_i64(
                            black_box(&backend),
                            black_box(&dev_data),
                            black_box(&dev_offsets),
                            black_box(&mut dev_out),
                            k as i64,
                        )
                        .expect("HIP reduce_argmin_i64 failed");
                        backend.download_slice(&dev_out, black_box(&mut out_host));
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

// ── reduce_count ──────────────────────────────────────────────────────────────
//
// CPU uses `reduce_count_64(toptr, parents)` — parents, not offsets.
// HIP uses `segmented_count(offsets, out, n_segments)` — offsets only, no data.

fn bench_reduce_count(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::count::reduce_count_64;

    #[cfg(all(feature = "hip", hip_rocm))]
    use rawkward::kernels::hip::reduce::count::segmented_count;

    #[cfg(all(feature = "hip", hip_rocm))]
    let backend = HipBackend::new();

    let mut group = c.benchmark_group("reduce_count_hip");

    for &(n, k) in SIZES {
        let parents = make_parents_contig(n, k);
        let offsets = make_offsets_contig(n, k);
        let label = format!("{n}/{k}");
        group.throughput(Throughput::Elements(n as u64));

        let mut out_cpu = vec![0i64; k];
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter_batched_ref(
                || (),
                |_| reduce_count_64(black_box(&mut out_cpu), black_box(&parents)),
                BatchSize::SmallInput,
            )
        });

        #[cfg(all(feature = "hip", hip_rocm))]
        {
            let dev_offsets = backend.upload_slice::<i64>(&offsets);
            let mut dev_out = unsafe { backend.alloc_slice::<i64>(k) };
            let mut out_host = vec![0i64; k];

            group.bench_with_input(BenchmarkId::new("hip", &label), &(), |b, _| {
                b.iter_batched_ref(
                    || (),
                    |_| {
                        segmented_count(
                            black_box(&backend),
                            black_box(&dev_offsets),
                            black_box(&mut dev_out),
                            k as i64,
                        )
                        .expect("HIP reduce_count failed");
                        backend.download_slice(&dev_out, black_box(&mut out_host));
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

// ── reduce_countnonzero ───────────────────────────────────────────────────────
//
// CPU uses `reduce_countnonzero(toptr, fromptr, parents)` — parents.
// HIP uses `segmented_countnonzero_i64(data, offsets, out, n)` — offsets.

fn bench_reduce_countnonzero(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::countnonzero::reduce_countnonzero;

    #[cfg(all(feature = "hip", hip_rocm))]
    use rawkward::kernels::hip::reduce::countnonzero::segmented_countnonzero_i64;

    #[cfg(all(feature = "hip", hip_rocm))]
    let backend = HipBackend::new();

    let mut group = c.benchmark_group("reduce_countnonzero_hip_i64");

    for &(n, k) in SIZES {
        let data = make_data_i64(n);
        let parents = make_parents_contig(n, k);
        let offsets = make_offsets_contig(n, k);
        let label = format!("{n}/{k}");
        group.throughput(Throughput::Elements(n as u64));

        let mut out_cpu = vec![0i64; k];
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter_batched_ref(
                || (),
                |_| {
                    reduce_countnonzero::<i64>(
                        black_box(&mut out_cpu),
                        black_box(&data),
                        black_box(&parents),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        #[cfg(all(feature = "hip", hip_rocm))]
        {
            let dev_data = backend.upload_slice::<i64>(&data);
            let dev_offsets = backend.upload_slice::<i64>(&offsets);
            let mut dev_out = unsafe { backend.alloc_slice::<i64>(k) };
            let mut out_host = vec![0i64; k];

            group.bench_with_input(BenchmarkId::new("hip", &label), &(), |b, _| {
                b.iter_batched_ref(
                    || (),
                    |_| {
                        segmented_countnonzero_i64(
                            black_box(&backend),
                            black_box(&dev_data),
                            black_box(&dev_offsets),
                            black_box(&mut dev_out),
                            k as i64,
                        )
                        .expect("HIP reduce_countnonzero_i64 failed");
                        backend.download_slice(&dev_out, black_box(&mut out_host));
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

// ── reduce_max ────────────────────────────────────────────────────────────────

fn bench_reduce_max(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::max::reduce_max;

    #[cfg(all(feature = "hip", hip_rocm))]
    use rawkward::kernels::hip::reduce::max::segmented_max_i64;

    #[cfg(all(feature = "hip", hip_rocm))]
    let backend = HipBackend::new();

    let mut group = c.benchmark_group("reduce_max_hip_i64");

    for &(n, k) in SIZES {
        let data = make_data_i64(n);
        let offsets = make_offsets_contig(n, k);
        let label = format!("{n}/{k}");
        group.throughput(Throughput::Elements(n as u64));

        let mut out_cpu = vec![0i64; k];
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter_batched_ref(
                || (),
                |_| {
                    reduce_max::<i64, i64>(
                        black_box(&mut out_cpu),
                        black_box(&data),
                        black_box(&offsets),
                        i64::MIN,
                    )
                },
                BatchSize::SmallInput,
            )
        });

        #[cfg(all(feature = "hip", hip_rocm))]
        {
            let dev_data = backend.upload_slice::<i64>(&data);
            let dev_offsets = backend.upload_slice::<i64>(&offsets);
            let mut dev_out = unsafe { backend.alloc_slice::<i64>(k) };
            let mut out_host = vec![0i64; k];

            group.bench_with_input(BenchmarkId::new("hip", &label), &(), |b, _| {
                b.iter_batched_ref(
                    || (),
                    |_| {
                        segmented_max_i64(
                            black_box(&backend),
                            black_box(&dev_data),
                            black_box(&dev_offsets),
                            black_box(&mut dev_out),
                            k as i64,
                        )
                        .expect("HIP reduce_max_i64 failed");
                        backend.download_slice(&dev_out, black_box(&mut out_host));
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

// ── reduce_min ────────────────────────────────────────────────────────────────

fn bench_reduce_min(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::min::reduce_min;

    #[cfg(all(feature = "hip", hip_rocm))]
    use rawkward::kernels::hip::reduce::min::segmented_min_i64;

    #[cfg(all(feature = "hip", hip_rocm))]
    let backend = HipBackend::new();

    let mut group = c.benchmark_group("reduce_min_hip_i64");

    for &(n, k) in SIZES {
        let data = make_data_i64(n);
        let offsets = make_offsets_contig(n, k);
        let label = format!("{n}/{k}");
        group.throughput(Throughput::Elements(n as u64));

        let mut out_cpu = vec![0i64; k];
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter_batched_ref(
                || (),
                |_| {
                    reduce_min::<i64, i64>(
                        black_box(&mut out_cpu),
                        black_box(&data),
                        black_box(&offsets),
                        i64::MAX,
                    )
                },
                BatchSize::SmallInput,
            )
        });

        #[cfg(all(feature = "hip", hip_rocm))]
        {
            let dev_data = backend.upload_slice::<i64>(&data);
            let dev_offsets = backend.upload_slice::<i64>(&offsets);
            let mut dev_out = unsafe { backend.alloc_slice::<i64>(k) };
            let mut out_host = vec![0i64; k];

            group.bench_with_input(BenchmarkId::new("hip", &label), &(), |b, _| {
                b.iter_batched_ref(
                    || (),
                    |_| {
                        segmented_min_i64(
                            black_box(&backend),
                            black_box(&dev_data),
                            black_box(&dev_offsets),
                            black_box(&mut dev_out),
                            k as i64,
                        )
                        .expect("HIP reduce_min_i64 failed");
                        backend.download_slice(&dev_out, black_box(&mut out_host));
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

// ── reduce_prod ───────────────────────────────────────────────────────────────

fn bench_reduce_prod(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::prod::reduce_prod;

    #[cfg(all(feature = "hip", hip_rocm))]
    use rawkward::kernels::hip::reduce::prod::segmented_prod_i64;

    #[cfg(all(feature = "hip", hip_rocm))]
    let backend = HipBackend::new();

    let mut group = c.benchmark_group("reduce_prod_hip_i64");

    for &(n, k) in SIZES {
        let data = make_data_i64(n);
        let offsets = make_offsets_contig(n, k);
        let label = format!("{n}/{k}");
        group.throughput(Throughput::Elements(n as u64));

        let mut out_cpu = vec![0i64; k];
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter_batched_ref(
                || (),
                |_| {
                    reduce_prod::<i64, i64>(
                        black_box(&mut out_cpu),
                        black_box(&data),
                        black_box(&offsets),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        #[cfg(all(feature = "hip", hip_rocm))]
        {
            let dev_data = backend.upload_slice::<i64>(&data);
            let dev_offsets = backend.upload_slice::<i64>(&offsets);
            let mut dev_out = unsafe { backend.alloc_slice::<i64>(k) };
            let mut out_host = vec![0i64; k];

            group.bench_with_input(BenchmarkId::new("hip", &label), &(), |b, _| {
                b.iter_batched_ref(
                    || (),
                    |_| {
                        segmented_prod_i64(
                            black_box(&backend),
                            black_box(&dev_data),
                            black_box(&dev_offsets),
                            black_box(&mut dev_out),
                            k as i64,
                        )
                        .expect("HIP reduce_prod_i64 failed");
                        backend.download_slice(&dev_out, black_box(&mut out_host));
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

// ── reduce_sum ────────────────────────────────────────────────────────────────

fn bench_reduce_sum(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::sum::reduce_sum;

    #[cfg(all(feature = "hip", hip_rocm))]
    use rawkward::kernels::hip::reduce::sum::segmented_sum_i64;

    #[cfg(all(feature = "hip", hip_rocm))]
    let backend = HipBackend::new();

    let mut group = c.benchmark_group("reduce_sum_hip_i64");

    for &(n, k) in SIZES {
        let data = make_data_i64(n);
        let offsets = make_offsets_contig(n, k);
        let label = format!("{n}/{k}");
        group.throughput(Throughput::Elements(n as u64));

        let mut out_cpu = vec![0i64; k];
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter_batched_ref(
                || (),
                |_| {
                    reduce_sum::<i64, i64>(
                        black_box(&mut out_cpu),
                        black_box(&data),
                        black_box(&offsets),
                    )
                },
                BatchSize::SmallInput,
            )
        });

        #[cfg(all(feature = "hip", hip_rocm))]
        {
            let dev_data = backend.upload_slice::<i64>(&data);
            let dev_offsets = backend.upload_slice::<i64>(&offsets);
            let mut dev_out = unsafe { backend.alloc_slice::<i64>(k) };
            let mut out_host = vec![0i64; k];

            group.bench_with_input(BenchmarkId::new("hip", &label), &(), |b, _| {
                b.iter_batched_ref(
                    || (),
                    |_| {
                        segmented_sum_i64(
                            black_box(&backend),
                            black_box(&dev_data),
                            black_box(&dev_offsets),
                            black_box(&mut dev_out),
                            k as i64,
                        )
                        .expect("HIP reduce_sum_i64 failed");
                        backend.download_slice(&dev_out, black_box(&mut out_host));
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

// ── Wire up criterion ─────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_reduce_argmax,
    bench_reduce_argmin,
    bench_reduce_count,
    bench_reduce_countnonzero,
    bench_reduce_max,
    bench_reduce_min,
    bench_reduce_prod,
    bench_reduce_sum,
);
criterion_main!(benches);
