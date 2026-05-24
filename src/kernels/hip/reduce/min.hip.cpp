// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause
#include <hip/hip_runtime.h>
#include <limits>

template <typename T>
__device__ inline void segmented_min_body(
    const T* data, const long long* offsets, long long* out,
    long long n_segments, long long seg)
{
    if (seg >= n_segments) return;
    long long start = offsets[seg], end = offsets[seg + 1];
    if (end <= start) { out[seg] = static_cast<long long>(std::numeric_limits<T>::max()); return; }
    T acc = std::numeric_limits<T>::max();
    for (long long i = start; i < end; ++i) {
        T v = data[i];
        if (v < acc) acc = v;
    }
    out[seg] = static_cast<long long>(acc);
}

extern "C" __global__ void segmented_min_f32(
    const float* data, const long long* offsets, long long* out, long long n_segments)
{ long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  segmented_min_body(data, offsets, out, n_segments, seg); }

extern "C" __global__ void segmented_min_f64(
    const double* data, const long long* offsets, long long* out, long long n_segments)
{ long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  segmented_min_body(data, offsets, out, n_segments, seg); }

extern "C" __global__ void segmented_min_i32(
    const int* data, const long long* offsets, long long* out, long long n_segments)
{ long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  segmented_min_body(data, offsets, out, n_segments, seg); }

extern "C" __global__ void segmented_min_i64(
    const long long* data, const long long* offsets, long long* out, long long n_segments)
{ long long seg = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  segmented_min_body(data, offsets, out, n_segments, seg); }
