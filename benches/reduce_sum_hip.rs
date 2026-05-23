// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

//! Segmented reduce-sum: CPU Rust scalar vs HIP GPU (della-milan).
//!
//! Input sizes and data-generation match `benches/reduce_sum_metal.rs` exactly
//! so the numbers are directly comparable across the two machines.
//!
//! # Run
//!
//! ```text
//! cargo bench --features hip --bench reduce_sum_hip
//! ```
//!
//! # Run under rocprof (per-kernel duration + memory bandwidth)
//!
//! ```text
//! rocprof --stats --output-dir rocprof_out \
//!     cargo bench --features hip --bench reduce_sum_hip
//! ```
//!
//! Results land in `rocprof_out/results.stats.csv`.
//!
//! # What is measured
//!
//! ```text
//! cpu       — pure in-process scalar reduction on already-resident CPU data.
//!             Matches the `cpu` group in reduce_sum_metal.rs.
//!
//! gpu_rnd   — full round-trip per iteration: H2D upload + kernel dispatch
//!             + D2H download.  Models a caller with no persistent device
//!             state (cold-start cost).  Equivalent to `metal_alloc` but
//!             with PCIe rather than unified-memory transfers.
//!
//! gpu_disp  — device buffers pre-uploaded once; per-iteration cost is
//!             kernel dispatch + D2H download only (no H2D).  Models a
//!             caller that keeps data resident on the device across calls.
//!             Note: D2H is always included here because there is no unified
//!             memory on a discrete GPU — the result must cross PCIe to reach
//!             the CPU.  The nearest Metal analogue is `metal_unified`
//!             (persistent buffers), though Metal avoids the D2H transfer
//!             entirely via StorageModeShared.
//! ```
//!
//! # Input sizes
//!
//! ```text
//!   small   1 K elems,  128 segments  (avg  8 elems/seg)
//!   medium 64 K elems, 4 K segments   (avg 16 elems/seg)
//!   large   1 M elems, 16 K segments  (avg 64 elems/seg)
//! ```

#[cfg(all(feature = "hip", hip_rocm))]
use std::time::{Duration, Instant};

#[cfg(all(feature = "hip", hip_rocm))]
use rawkward::backend::GpuBackend;
#[cfg(all(feature = "hip", hip_rocm))]
use rawkward::backend::hip::HipBackend;

#[cfg(all(feature = "hip", hip_rocm))]
use rawkward::kernels::cpu::reduce::sum::{
    reduce_sum_float32_float32_64, reduce_sum_int64_int64_64,
};
#[cfg(all(feature = "hip", hip_rocm))]
use rawkward::kernels::hip::reduce::sum::{segmented_sum_f32, segmented_sum_i64};

// ── Input sizes ───────────────────────────────────────────────────────────────
// Mirror `SIZES` in `benches/reduce_sum_metal.rs`.

#[cfg(all(feature = "hip", hip_rocm))]
const SIZES: &[(&str, usize, usize)] = &[
    ("small   1K/128", 1_024, 128),
    ("medium 64K/4K", 65_536, 4_096),
    ("large   1M/16K", 1_048_576, 16_384),
];

// ── Data generation ───────────────────────────────────────────────────────────
// Identical LCG to `benches/reduce_sum_metal.rs` — same seeds, same output.

#[cfg(all(feature = "hip", hip_rocm))]
fn make_offsets_contig(lenparents: usize, outlength: usize) -> Vec<i64> {
    let group_size = lenparents.div_ceil(outlength).max(1);
    (0..=outlength)
        .map(|g| ((g * group_size).min(lenparents)) as i64)
        .collect()
}

#[cfg(all(feature = "hip", hip_rocm))]
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

#[cfg(all(feature = "hip", hip_rocm))]
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

// ── Timing helper ─────────────────────────────────────────────────────────────

#[cfg(all(feature = "hip", hip_rocm))]
fn time_iters<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> Duration {
    for _ in 0..warmup {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed() / iters as u32
}

// ── Table printer ─────────────────────────────────────────────────────────────

#[cfg(all(feature = "hip", hip_rocm))]
fn print_row(label: &str, cpu: Duration, rnd: Duration, disp: Duration) {
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    let ratio = |base: Duration, gpu: Duration| base.as_secs_f64() / gpu.as_secs_f64();
    println!(
        "{:<18} {:>10.2} {:>12.2} {:>12.2} {:>8.2}x {:>8.2}x",
        label,
        us(cpu),
        us(rnd),
        us(disp),
        ratio(cpu, rnd),
        ratio(cpu, disp),
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    #[cfg(not(all(feature = "hip", hip_rocm)))]
    eprintln!(
        "Build with --features hip on a ROCm machine to enable GPU benchmarks.\n\
         CPU-only: cargo bench --bench reduce_sum_hip  (prints nothing useful here)."
    );

    #[cfg(all(feature = "hip", hip_rocm))]
    run();
}

#[cfg(all(feature = "hip", hip_rocm))]
fn run() {
    let backend = HipBackend::new();

    // More iterations than argsort bench; reduce-sum is fast so variance is
    // higher relative to timer resolution at small sizes.
    const WARMUP: usize = 10;
    const ITERS: usize = 100;

    let hdr = || {
        println!(
            "{:<18} {:>10} {:>12} {:>12} {:>9} {:>9}",
            "size", "cpu (µs)", "gpu_rnd (µs)", "gpu_disp (µs)", "rnd/cpu", "disp/cpu"
        );
        println!("{}", "─".repeat(76));
    };

    // ── f32 ──────────────────────────────────────────────────────────────────

    println!("\n── reduce_sum f32 ────────────────────────────────────────────────────");
    hdr();

    for &(label, n, k) in SIZES {
        let data = make_data_f32(n);
        let offsets = make_offsets_contig(n, k);

        let mut out_cpu = vec![0.0f32; k];
        let mut out_host = vec![0.0f32; k];

        // ── cpu ────────────────────────────────────────────────────────────
        let cpu = time_iters(
            || reduce_sum_float32_float32_64(&mut out_cpu, &data, &offsets),
            WARMUP,
            ITERS,
        );

        // ── gpu_rnd: full round-trip per iteration ────────────────────────
        // H2D upload + kernel dispatch + D2H download; hipFree on drop.
        let gpu_rnd = time_iters(
            || {
                let dev_data = backend.upload_slice(&data);
                let dev_offsets = backend.upload_slice(&offsets);
                let mut dev_out = unsafe { backend.alloc_slice::<f32>(k) };
                segmented_sum_f32(&backend, &dev_data, &dev_offsets, &mut dev_out, k as i64)
                    .expect("HIP reduce_sum_f32 round-trip failed");
                // hipMemcpy (synchronous on null stream) — ordered after the
                // async kernel launch, so no explicit hipDeviceSynchronize needed.
                backend.download_slice(&dev_out, &mut out_host);
                // DevSlices drop here → hipFree
            },
            WARMUP,
            ITERS,
        );

        // ── gpu_disp: pre-upload; per-iteration: dispatch + D2H ──────────
        let dev_data = backend.upload_slice(&data);
        let dev_offsets = backend.upload_slice(&offsets);
        let mut dev_out = unsafe { backend.alloc_slice::<f32>(k) };

        let gpu_disp = time_iters(
            || {
                segmented_sum_f32(&backend, &dev_data, &dev_offsets, &mut dev_out, k as i64)
                    .expect("HIP reduce_sum_f32 dispatch failed");
                backend.download_slice(&dev_out, &mut out_host);
            },
            WARMUP,
            ITERS,
        );

        print_row(label, cpu, gpu_rnd, gpu_disp);
    }

    // ── i64 ──────────────────────────────────────────────────────────────────

    println!("\n── reduce_sum i64 ────────────────────────────────────────────────────");
    hdr();

    for &(label, n, k) in SIZES {
        let data = make_data_i64(n);
        let offsets = make_offsets_contig(n, k);

        let mut out_cpu = vec![0i64; k];
        let mut out_host = vec![0i64; k];

        // ── cpu ────────────────────────────────────────────────────────────
        let cpu = time_iters(
            || reduce_sum_int64_int64_64(&mut out_cpu, &data, &offsets),
            WARMUP,
            ITERS,
        );

        // ── gpu_rnd ────────────────────────────────────────────────────────
        let gpu_rnd = time_iters(
            || {
                let dev_data = backend.upload_slice(&data);
                let dev_offsets = backend.upload_slice(&offsets);
                let mut dev_out = unsafe { backend.alloc_slice::<i64>(k) };
                segmented_sum_i64(&backend, &dev_data, &dev_offsets, &mut dev_out, k as i64)
                    .expect("HIP reduce_sum_i64 round-trip failed");
                backend.download_slice(&dev_out, &mut out_host);
            },
            WARMUP,
            ITERS,
        );

        // ── gpu_disp ───────────────────────────────────────────────────────
        let dev_data = backend.upload_slice(&data);
        let dev_offsets = backend.upload_slice(&offsets);
        let mut dev_out = unsafe { backend.alloc_slice::<i64>(k) };

        let gpu_disp = time_iters(
            || {
                segmented_sum_i64(&backend, &dev_data, &dev_offsets, &mut dev_out, k as i64)
                    .expect("HIP reduce_sum_i64 dispatch failed");
                backend.download_slice(&dev_out, &mut out_host);
            },
            WARMUP,
            ITERS,
        );

        print_row(label, cpu, gpu_rnd, gpu_disp);
    }

    // ── Legend ────────────────────────────────────────────────────────────────

    println!();
    println!(
        "cpu      = scalar reduction, data already in CPU cache/memory.\n\
         gpu_rnd  = H2D + kernel + D2H per iteration  (cold-start / no persistent state).\n\
         gpu_disp = buffers pre-uploaded; kernel + D2H per iteration.\n\
         rnd/cpu  = cpu_time / gpu_rnd_time  (>1 means GPU is faster).\n\
         disp/cpu = cpu_time / gpu_disp_time (>1 means GPU is faster).\n"
    );
    println!(
        "To profile GPU kernel statistics:\n\
         \n  \
         rocprof --stats --output-dir rocprof_out \\\n  \
             cargo bench --features hip --bench reduce_sum_hip\n\
         \n\
         Results → rocprof_out/results.stats.csv\n\
         Key columns: DurationNs (kernel time), FETCH_SIZE, WRITE_SIZE (bandwidth)."
    );
}
