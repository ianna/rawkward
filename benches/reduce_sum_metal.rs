// benches/reduce_sum_metal.rs
//
// Side-by-side benchmarks: Rust CPU scalar kernel vs Rust Metal GPU kernel for
// segmented reduce-sum.  The Metal side is compiled only on macOS; on other
// platforms those benchmark functions are no-ops so the file compiles cleanly
// everywhere.
//
// # Run
//
//   # All groups:
//   cargo bench --bench reduce_sum_metal
//
//   # Single group:
//   cargo bench --bench reduce_sum_metal -- reduce_sum_f32
//   cargo bench --bench reduce_sum_metal -- reduce_sum_i64
//
// # What is measured
//
//   cpu    — pure in-process scalar reduction on already-resident CPU data.
//   metal  — GPU dispatch round-trip: encode + commit + wait_until_completed.
//            Input data is uploaded ONCE before the bench loop; only kernel
//            execution (and Metal command-buffer overhead) is timed.
//
// This split isolates GPU compute throughput from data-transfer cost.  In a
// real pipeline on Apple Silicon the upload is free for the initial jagged
// array because StorageModeShared gives zero-copy access; the numbers here
// are therefore representative of steady-state throughput.
//
// # Input sizes
//
// Three sizes matching `benches/kernels.rs`:
//
//   small   1 K elems,  128 segments  (avg  8 elems/seg)   — call-overhead regime
//   medium 64 K elems, 4 K segments   (avg 16 elems/seg)   — cache-resident
//   large   1 M elems, 16 K segments  (avg 64 elems/seg)   — memory-bandwidth regime

#![allow(clippy::too_many_arguments)]

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};

// ── Input sizes ───────────────────────────────────────────────────────────────

/// `(n_elements, n_segments)` — mirrors `SIZES` in `benches/kernels.rs`.
const SIZES: &[(usize, usize)] = &[
    (1_024, 128),        // small:  1 K elements, 128 segments  (avg  8 elems/seg)
    (65_536, 4_096),     // medium: 64 K elements, 4 K segments (avg 16 elems/seg)
    (1_048_576, 16_384), // large:  1 M elements, 16 K segments (avg 64 elems/seg)
];

// ── Data generation ───────────────────────────────────────────────────────────

/// Contiguous offsets: `offsets[g]..offsets[g+1]` is group `g`.
/// Length is `outlength + 1`; `offsets[outlength] == lenparents`.
fn make_offsets_contig(lenparents: usize, outlength: usize) -> Vec<i64> {
    let group_size = lenparents.div_ceil(outlength).max(1);
    (0..=outlength)
        .map(|g| ((g * group_size).min(lenparents)) as i64)
        .collect()
}

/// Deterministic `f32` data via LCG — reproducible across runs.
fn make_data_f32(n: usize) -> Vec<f32> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to [-1, 1] so partial sums stay well-behaved.
            ((state >> 33) as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

/// Deterministic `i64` data via LCG — reproducible across runs.
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

// ── Bench-pair macro ──────────────────────────────────────────────────────────
//
// Adds a "cpu" entry and (on macOS) a "metal" entry to the same
// `BenchmarkGroup` so they appear side-by-side in criterion's output and
// comparison reports.

macro_rules! bench_pair {
    ($group:expr, $label:expr, $cpu:expr, $metal:expr $(,)?) => {{
        let group = &mut $group;
        group.bench_with_input(BenchmarkId::new("cpu", $label), &(), |b, _| {
            b.iter_batched_ref(|| (), |_| $cpu, BatchSize::SmallInput)
        });
        #[cfg(target_os = "macos")]
        group.bench_with_input(BenchmarkId::new("metal", $label), &(), |b, _| {
            b.iter_batched_ref(|| (), |_| $metal, BatchSize::SmallInput)
        });
        // Suppress "unused" warnings on non-macOS.
        #[cfg(not(target_os = "macos"))]
        let _ = $metal;
    }};
}

// ── reduce_sum f32 ────────────────────────────────────────────────────────────

fn bench_reduce_sum_f32(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::sum::reduce_sum_float32_float32_64;

    // Metal backend initialised once — shader compilation happens here and is
    // NOT included in any per-iteration measurement.
    #[cfg(target_os = "macos")]
    let backend = {
        use rawkward::backend::metal::MetalBackend;
        match MetalBackend::new() {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("reduce_sum_f32: Metal unavailable ({e:?}), skipping GPU bench");
                None
            }
        }
    };

    let mut group = c.benchmark_group("reduce_sum_f32");

    for &(n, k) in SIZES {
        let data    = make_data_f32(n);
        let offsets = make_offsets_contig(n, k);
        let label   = format!("{n}/{k}");

        group.throughput(Throughput::Elements(n as u64));

        // CPU: allocate output buffer once; reuse across iterations.
        let mut out_cpu = vec![0.0f32; k];

        // Metal: upload input buffers once; allocate output buffer once.
        // The closures below capture these by reference so nothing is re-uploaded
        // per iteration.
        #[cfg(target_os = "macos")]
        let (data_dev, offsets_dev, mut out_dev) = if let Some(ref b) = backend {
            use rawkward::backend::GpuBackend;
            let d = b.upload_slice(&data);
            let o = b.upload_slice(&offsets);
            let out = unsafe { b.alloc_slice::<f32>(k) };
            (Some(d), Some(o), Some(out))
        } else {
            (None, None, None)
        };

        bench_pair!(
            group,
            &label,
            // ── CPU ──────────────────────────────────────────────────────────
            reduce_sum_float32_float32_64(
                black_box(&mut out_cpu),
                black_box(&data),
                black_box(&offsets),
            ),
            // ── Metal ────────────────────────────────────────────────────────
            // Measures: encode + commit + wait_until_completed.
            // Data is already in a StorageModeShared buffer — no transfer cost.
            {
                #[cfg(target_os = "macos")]
                if let (Some(ref b), Some(ref d), Some(ref o), Some(ref mut out)) =
                    (backend.as_ref(), data_dev.as_ref(), offsets_dev.as_ref(), out_dev.as_mut())
                {
                    use rawkward::kernels::metal::reduce::sum::segmented_sum_f32;
                    unsafe {
                        segmented_sum_f32(
                            black_box(b),
                            black_box(d),
                            black_box(o),
                            black_box(out),
                            k as u64,
                        )
                        .expect("Metal reduce_sum_f32 failed");
                    }
                }
            },
        );
    }

    group.finish();
}

// ── reduce_sum i64 ────────────────────────────────────────────────────────────

fn bench_reduce_sum_i64(c: &mut Criterion) {
    use rawkward::kernels::cpu::reduce::sum::reduce_sum_int64_int64_64;

    #[cfg(target_os = "macos")]
    let backend = {
        use rawkward::backend::metal::MetalBackend;
        match MetalBackend::new() {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("reduce_sum_i64: Metal unavailable ({e:?}), skipping GPU bench");
                None
            }
        }
    };

    let mut group = c.benchmark_group("reduce_sum_i64");

    for &(n, k) in SIZES {
        let data    = make_data_i64(n);
        let offsets = make_offsets_contig(n, k);
        let label   = format!("{n}/{k}");

        group.throughput(Throughput::Elements(n as u64));

        let mut out_cpu = vec![0i64; k];

        #[cfg(target_os = "macos")]
        let (data_dev, offsets_dev, mut out_dev) = if let Some(ref b) = backend {
            use rawkward::backend::GpuBackend;
            let d = b.upload_slice(&data);
            let o = b.upload_slice(&offsets);
            let out = unsafe { b.alloc_slice::<i64>(k) };
            (Some(d), Some(o), Some(out))
        } else {
            (None, None, None)
        };

        bench_pair!(
            group,
            &label,
            // ── CPU ──────────────────────────────────────────────────────────
            reduce_sum_int64_int64_64(
                black_box(&mut out_cpu),
                black_box(&data),
                black_box(&offsets),
            ),
            // ── Metal ────────────────────────────────────────────────────────
            {
                #[cfg(target_os = "macos")]
                if let (Some(ref b), Some(ref d), Some(ref o), Some(ref mut out)) =
                    (backend.as_ref(), data_dev.as_ref(), offsets_dev.as_ref(), out_dev.as_mut())
                {
                    use rawkward::kernels::metal::reduce::sum::segmented_sum_i64;
                    unsafe {
                        segmented_sum_i64(
                            black_box(b),
                            black_box(d),
                            black_box(o),
                            black_box(out),
                            k as u64,
                        )
                        .expect("Metal reduce_sum_i64 failed");
                    }
                }
            },
        );
    }

    group.finish();
}

// ── round-trip bench (upload + dispatch + download) ───────────────────────────
//
// Measures the FULL cost of using the GPU from a CPU-side caller that does not
// keep data persistently resident on the device.  Useful for understanding the
// break-even point: at which problem size does GPU compute outweigh the
// round-trip overhead?
//
// Only meaningful on macOS; the CPU side is omitted here since its cost is
// already captured in `bench_reduce_sum_f32`.

#[cfg(target_os = "macos")]
fn bench_reduce_sum_f32_roundtrip(c: &mut Criterion) {
    use rawkward::backend::{GpuBackend, metal::MetalBackend};
    use rawkward::kernels::metal::reduce::sum::segmented_sum_f32;

    let backend = match MetalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("reduce_sum_f32_roundtrip: Metal unavailable ({e:?}), skipping");
            return;
        }
    };

    let mut group = c.benchmark_group("reduce_sum_f32_roundtrip");

    for &(n, k) in SIZES {
        let data    = make_data_f32(n);
        let offsets = make_offsets_contig(n, k);
        let label   = format!("{n}/{k}");

        group.throughput(Throughput::Elements(n as u64));

        let mut out_host = vec![0.0f32; k];

        group.bench_with_input(BenchmarkId::new("metal", &label), &(), |b, _| {
            b.iter_batched_ref(
                || (),
                |_| {
                    // upload
                    let data_dev    = backend.upload_slice(black_box(&data));
                    let offsets_dev = backend.upload_slice(black_box(&offsets));
                    let mut out_dev = unsafe { backend.alloc_slice::<f32>(k) };
                    // dispatch + wait
                    unsafe {
                        segmented_sum_f32(
                            &backend,
                            &data_dev,
                            &offsets_dev,
                            &mut out_dev,
                            k as u64,
                        )
                        .expect("Metal reduce_sum_f32 roundtrip failed");
                    }
                    // download
                    backend.download_slice(&out_dev, black_box(&mut out_host));
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

#[cfg(not(target_os = "macos"))]
fn bench_reduce_sum_f32_roundtrip(_c: &mut Criterion) {}

// ── Wire up criterion ─────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_reduce_sum_f32,
    bench_reduce_sum_i64,
    bench_reduce_sum_f32_roundtrip,
);
criterion_main!(benches);
