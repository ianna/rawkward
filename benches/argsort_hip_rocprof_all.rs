// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! CPU vs GPU argsort comparison benchmark.
//!
//! Run as a plain timing table:
//!   cargo bench --features hip --bench argsort_hip_rocprof
//!
//! Run under rocprof for GPU kernel statistics:
//!   rocprof --stats --output-dir rocprof_out \
//!     cargo bench --features hip --bench argsort_hip_rocprof
//!
//! rocprof writes per-kernel duration, call counts, and memory
//! bandwidth to rocprof_out/results.stats.csv.
//!
//! For hardware counter profiling (VGPR usage, cache efficiency):
//!   rocprof --stats --timestamp on -i counters.txt --output-dir rocprof_out \
//!     cargo bench --features hip --bench argsort_hip_rocprof
//! where counters.txt lists metric names, e.g.:
//!   pmc: TCC_HIT_sum, TCC_MISS_sum, FETCH_SIZE, WRITE_SIZE

use rand::Rng;
use rayon::prelude::*;
use std::time::{Duration, Instant};

#[cfg(feature = "hip")]
use rawkward::backend::hip::HipBackend;
#[cfg(feature = "hip")]
use rawkward::backend::{GpuBackend, GpuError};
#[cfg(feature = "hip")]
use rawkward::kernels::hip::sort::argsort::{argsort_large, argsort_medium, argsort_small};

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

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
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Single-threaded argsort over a jagged array.
fn cpu_argsort_serial(values: &[f32], offsets: &[i64], out: &mut [i64]) {
    let nlists = offsets.len() - 1;
    for i in 0..nlists {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        let mut indices: Vec<i64> = (start as i64..end as i64).collect();
        indices.sort_unstable_by(|&a, &b| {
            values[a as usize]
                .partial_cmp(&values[b as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out[start..end].copy_from_slice(&indices);
    }
}

/// Rayon parallel argsort — one list per thread-pool task.
///
/// Sorts each list independently in parallel, collecting results into
/// intermediate Vecs before copying back. The copy is O(n) serial but
/// sort time dominates for any non-trivial list length.
fn cpu_argsort_parallel(values: &[f32], offsets: &[i64], out: &mut [i64]) {
    let nlists = offsets.len() - 1;

    let sorted: Vec<(usize, Vec<i64>)> = (0..nlists)
        .into_par_iter()
        .map(|i| {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            let mut indices: Vec<i64> = (start as i64..end as i64).collect();
            indices.sort_unstable_by(|&a, &b| {
                values[a as usize]
                    .partial_cmp(&values[b as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            (start, indices)
        })
        .collect();

    for (start, indices) in sorted {
        out[start..start + indices.len()].copy_from_slice(&indices);
    }
}

// ---------------------------------------------------------------------------
// GPU dispatch
// ---------------------------------------------------------------------------

/// Full-round-trip GPU argsort: H2D upload → kernel dispatch → D2H download.
///
/// Kernels are resolved once before the per-list loop. Lists are dispatched
/// to the appropriate kernel tier based on length:
///   ≤ 64    → argsort_small_hip  (in-register bitonic)
///   ≤ 256   → argsort_medium_hip (LDS bitonic)
///   > 256   → argsort_large_hip  (rocPRIM segmented)
///
/// TODO: small and medium lists should be batched into a single kernel
/// launch each (one thread block per list, grid = n_lists_in_tier).
/// Currently each list is a separate launch, which is correct but
/// sub-optimal for large list counts.
#[cfg(feature = "hip")]
fn gpu_argsort_jagged<B: GpuBackend>(
    backend: &B,
    values: &[f32],
    offsets: &[i64],
    out: &mut [i64],
) -> Result<(), GpuError> {
    let dev_values = backend.upload_slice(values);
    let dev_offsets = backend.upload_slice(offsets);
    let mut dev_out = unsafe { backend.alloc_slice::<i64>(values.len()) };

    let k_small = backend
        .get_kernel("argsort_small_hip")
        .map_err(GpuError::HipError)?;
    let k_medium = backend
        .get_kernel("argsort_medium_hip")
        .map_err(GpuError::HipError)?;
    let k_large = backend
        .get_kernel("argsort_large_hip")
        .map_err(GpuError::HipError)?;

    let nlists = offsets.len() - 1;

    // Collect large-list ids for the batched rocPRIM call.
    let mut has_large = false;

    for i in 0..nlists {
        let list_len = (offsets[i + 1] - offsets[i]) as usize;

        match list_len {
            0..=64 => {
                argsort_small(
                    backend,
                    &dev_values,
                    &dev_offsets,
                    &mut dev_out,
                    i as i64,
                    (1, 1, 1),
                    (64, 1, 1),
                )?;
            }
            65..=256 => {
                argsort_medium(
                    backend,
                    &dev_values,
                    &dev_offsets,
                    &mut dev_out,
                    i as i64,
                    (1, 1, 1),
                    (256, 1, 1),
                )?;
            }
            _ => {
                has_large = true;
            }
        }
    }

    // Large lists: one batched rocPRIM call covers all of them.
    if has_large {
        argsort_large(
            backend,
            &dev_values,
            &dev_offsets,
            &mut dev_out,
            values.len() as i64,
            nlists as i64,
            (256, 1, 1),
            (256, 1, 1),
        )?;
    }

    backend.download_slice(&dev_out, out);
    Ok(())
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

fn time_iters<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> Duration {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed() / iters as u32
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    #[cfg(not(feature = "hip"))]
    {
        eprintln!("Build with --features hip to enable GPU comparison.");
        return;
    }

    #[cfg(feature = "hip")]
    {
        let backend = HipBackend::new();

        // Dataset configs: (label, nlists, min_list_len, max_list_len)
        // Chosen to exercise the three kernel tiers:
        //   small  ≤ 64    → in-register bitonic sort
        //   medium ≤ 256   → LDS bitonic sort
        //   large  > 256   → rocPRIM segmented sort
        let configs: &[(&str, usize, usize, usize)] = &[
            ("small  (8–64)", 50_000, 8, 64),
            ("medium (65–256)", 20_000, 65, 256),
            ("large  (257–2k)", 5_000, 257, 2048),
            ("mixed  (8–2048)", 20_000, 8, 2048),
        ];

        const WARMUP: usize = 3;
        const ITERS: usize = 20;

        println!(
            "\n{:<22} {:>11} {:>11} {:>11} {:>9} {:>9}",
            "dataset", "cpu-ser(ms)", "cpu-par(ms)", "gpu-rnd(ms)", "ser/gpu", "par/gpu"
        );
        println!("{}", "─".repeat(77));

        for &(label, nlists, min_len, max_len) in configs {
            let (values, offsets) = generate_jagged(nlists, min_len, max_len);
            let mut out = vec![0i64; values.len()];

            let serial = time_iters(
                || cpu_argsort_serial(&values, &offsets, &mut out),
                WARMUP,
                ITERS,
            );

            let parallel = time_iters(
                || cpu_argsort_parallel(&values, &offsets, &mut out),
                WARMUP,
                ITERS,
            );

            // GPU time includes H2D transfer + kernel(s) + D2H transfer.
            let gpu = time_iters(
                || {
                    gpu_argsort_jagged(&backend, &values, &offsets, &mut out)
                        .expect("gpu_argsort_jagged failed");
                },
                WARMUP,
                ITERS,
            );

            let ms = |d: Duration| d.as_secs_f64() * 1000.0;
            let speedup = |b: Duration| b.as_secs_f64() / gpu.as_secs_f64();

            println!(
                "{:<22} {:>11.3} {:>11.3} {:>11.3} {:>8.2}x {:>8.2}x",
                label,
                ms(serial),
                ms(parallel),
                ms(gpu),
                speedup(serial),
                speedup(parallel),
            );
        }

        println!();
        println!("gpu-rnd = full round-trip (H2D + kernel + D2H).");
        println!(
            "par = rayon, {} logical cores.",
            rayon::current_num_threads()
        );
        println!();
        println!("To capture GPU kernel stats:");
        println!(
            "  rocprof --stats --output-dir rocprof_out \\\n    \
             cargo bench --features hip --bench argsort_hip_rocprof"
        );
        println!("Results in rocprof_out/results.stats.csv");
    }
}
