// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

// ── Segmented reduce-sum kernels ─────────────────────────────────────────────
//
// One thread handles one segment.  `thread_position_in_grid` is the segment
// index `g`; the thread sums data[offsets[g]..offsets[g+1]] into out[g].
//
// Launched via `dispatch_threads({ n_segments, 1, 1 }, { tg_size, 1, 1 })` so
// the runtime guarantees exactly n_segments threads run — no in-shader bounds
// check is required.
//
// Offsets use `long` (64-bit signed), matching the Rust `i64` type used by
// awkward-array throughout.  Data and output types are per-specialisation.
//
// For segments with many thousands of elements, a follow-on threadgroup-parallel
// reduction kernel (using threadgroup shared memory) would give better occupancy.
// This design is correct for all segment sizes and optimal when the segment
// count >> GPU threadgroup count (the typical awkward-array case).

// ── f32 ──────────────────────────────────────────────────────────────────────

kernel void reduce_sum_f32(
    device const float* data    [[ buffer(0) ]],
    device const long*  offsets [[ buffer(1) ]],
    device       float* out     [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    long  start = offsets[gid];
    long  stop  = offsets[gid + 1];
    float acc   = 0.0f;
    for (long i = start; i < stop; ++i) {
        acc += data[i];
    }
    out[gid] = acc;
}

// ── f64 ──────────────────────────────────────────────────────────────────────
// Metal does not support `double` (64-bit float) on any Apple GPU.  The
// `segmented_sum_f64` Rust function returns an error at runtime instead of
// dispatching an MSL kernel.  There is no `reduce_sum_f64` shader here.

// ── i32 ──────────────────────────────────────────────────────────────────────

kernel void reduce_sum_i32(
    device const int*  data    [[ buffer(0) ]],
    device const long* offsets [[ buffer(1) ]],
    device       int*  out     [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    long start = offsets[gid];
    long stop  = offsets[gid + 1];
    int  acc   = 0;
    for (long i = start; i < stop; ++i) {
        acc += data[i];
    }
    out[gid] = acc;
}

// ── i64 ──────────────────────────────────────────────────────────────────────

kernel void reduce_sum_i64(
    device const long* data    [[ buffer(0) ]],
    device const long* offsets [[ buffer(1) ]],
    device       long* out     [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    long start = offsets[gid];
    long stop  = offsets[gid + 1];
    long acc   = 0;
    for (long i = start; i < stop; ++i) {
        acc += data[i];
    }
    out[gid] = acc;
}
