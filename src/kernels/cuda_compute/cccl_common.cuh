// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause
//
// CCCL / CUB segmented-reduce helpers.
//
// This is the option-B path: instead of the hand-written per-segment kernels in
// `src/kernels/cuda/` (option A), it uses NVIDIA CUB's `DeviceSegmentedReduce`,
// mirroring how awkward-array's CUDA backend routes its reducers through
// `cuda.compute` (CCCL). Each `seg_*` template is a self-contained host
// orchestration: query CUB's temp-storage size, allocate it on the stream, run
// the reduction, free the scratch. The Rust side only hands us device pointers
// and a stream, exactly like awkward hands CuPy device buffers to CCCL.
//
// BUILD-HOST VERIFICATION: this file could not be compiled in the development
// environment (no CUDA toolkit / CUB headers / GPU). CUB ships inside the CUDA
// toolkit (`$CUDA_ROOT/include/cub`), so no extra dependency is needed, but the
// exact `DeviceSegmentedReduce` overload set and `KeyValuePair` layout must be
// confirmed against the CUB version on the target box. The control flow and CUB
// call shapes below follow the documented CUB API.

#pragma once

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdint>

namespace rawkward_cccl {

// Offsets are always signed 64-bit to match rawkward's `DevSlice<i64>` offset
// buffers (and awkward's Index64). CUB takes begin/end offset iterators, so we
// pass `d_off` and `d_off + 1`.
using offset_t = long long;

// CUB's segmented entry points historically take `int num_segments`. We cast
// and therefore support up to 2^31 - 1 segments per call; larger inputs must be
// chunked by the caller. (Newer CUB accepts 64-bit segment counts; if the
// target CUB provides that overload, widen `(int)n_seg` accordingly.)
static inline int seg_count_arg(offset_t n_seg) { return static_cast<int>(n_seg); }

// Multiply functor for the `prod` reduction (CUB has Sum/Min/Max but no Prod).
template <typename T>
struct MulOp {
    __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return a * b;
    }
};

// Maps any element to 0/1 for `countnonzero`, accumulated in i64.
struct NonZeroOp {
    template <typename T>
    __host__ __device__ __forceinline__ long long operator()(const T& x) const {
        return x != T(0) ? 1LL : 0LL;
    }
};

// Extract the `.key` (segment-local index) from CUB's arg-reduce KeyValuePair
// output into a plain i64 buffer, one entry per segment.
template <typename T>
__global__ void extract_keys(const cub::KeyValuePair<int, T>* pairs,
                             long long* out,
                             offset_t n_seg) {
    offset_t i = static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_seg) {
        out[i] = static_cast<long long>(pairs[i].key);
    }
}

// out[seg] = offsets[seg+1] - offsets[seg]. Trivial; no reduction needed.
__global__ void count_kernel(const offset_t* d_off, long long* out, offset_t n_seg) {
    offset_t i = static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_seg) {
        out[i] = d_off[i + 1] - d_off[i];
    }
}

// ---------------------------------------------------------------------------
// Sum / Min / Max — CUB provides dedicated entry points with correct identities
// ---------------------------------------------------------------------------

template <typename T>
inline cudaError_t seg_sum(const T* d_in, T* d_out, offset_t n_seg,
                           const offset_t* d_off, cudaStream_t s) {
    void* d_temp = nullptr;
    size_t bytes = 0;
    cudaError_t e = cub::DeviceSegmentedReduce::Sum(
        d_temp, bytes, d_in, d_out, seg_count_arg(n_seg), d_off, d_off + 1, s);
    if (e) return e;
    e = cudaMallocAsync(&d_temp, bytes, s);
    if (e) return e;
    e = cub::DeviceSegmentedReduce::Sum(
        d_temp, bytes, d_in, d_out, seg_count_arg(n_seg), d_off, d_off + 1, s);
    cudaError_t e_free = cudaFreeAsync(d_temp, s);
    return e ? e : e_free;
}

template <typename T>
inline cudaError_t seg_min(const T* d_in, T* d_out, offset_t n_seg,
                           const offset_t* d_off, cudaStream_t s) {
    void* d_temp = nullptr;
    size_t bytes = 0;
    cudaError_t e = cub::DeviceSegmentedReduce::Min(
        d_temp, bytes, d_in, d_out, seg_count_arg(n_seg), d_off, d_off + 1, s);
    if (e) return e;
    e = cudaMallocAsync(&d_temp, bytes, s);
    if (e) return e;
    e = cub::DeviceSegmentedReduce::Min(
        d_temp, bytes, d_in, d_out, seg_count_arg(n_seg), d_off, d_off + 1, s);
    cudaError_t e_free = cudaFreeAsync(d_temp, s);
    return e ? e : e_free;
}

template <typename T>
inline cudaError_t seg_max(const T* d_in, T* d_out, offset_t n_seg,
                           const offset_t* d_off, cudaStream_t s) {
    void* d_temp = nullptr;
    size_t bytes = 0;
    cudaError_t e = cub::DeviceSegmentedReduce::Max(
        d_temp, bytes, d_in, d_out, seg_count_arg(n_seg), d_off, d_off + 1, s);
    if (e) return e;
    e = cudaMallocAsync(&d_temp, bytes, s);
    if (e) return e;
    e = cub::DeviceSegmentedReduce::Max(
        d_temp, bytes, d_in, d_out, seg_count_arg(n_seg), d_off, d_off + 1, s);
    cudaError_t e_free = cudaFreeAsync(d_temp, s);
    return e ? e : e_free;
}

// ---------------------------------------------------------------------------
// Prod — generic Reduce with a multiply functor and identity 1
// ---------------------------------------------------------------------------

template <typename T>
inline cudaError_t seg_prod(const T* d_in, T* d_out, offset_t n_seg,
                            const offset_t* d_off, cudaStream_t s) {
    MulOp<T> op;
    void* d_temp = nullptr;
    size_t bytes = 0;
    cudaError_t e = cub::DeviceSegmentedReduce::Reduce(
        d_temp, bytes, d_in, d_out, seg_count_arg(n_seg), d_off, d_off + 1, op,
        T(1), s);
    if (e) return e;
    e = cudaMallocAsync(&d_temp, bytes, s);
    if (e) return e;
    e = cub::DeviceSegmentedReduce::Reduce(
        d_temp, bytes, d_in, d_out, seg_count_arg(n_seg), d_off, d_off + 1, op,
        T(1), s);
    cudaError_t e_free = cudaFreeAsync(d_temp, s);
    return e ? e : e_free;
}

// ---------------------------------------------------------------------------
// countnonzero — Sum over a transform iterator that maps element -> {0,1} (i64)
// ---------------------------------------------------------------------------

template <typename T>
inline cudaError_t seg_countnonzero(const T* d_in, long long* d_out,
                                    offset_t n_seg, const offset_t* d_off,
                                    cudaStream_t s) {
    cub::TransformInputIterator<long long, NonZeroOp, const T*> it(d_in, NonZeroOp{});
    void* d_temp = nullptr;
    size_t bytes = 0;
    cudaError_t e = cub::DeviceSegmentedReduce::Sum(
        d_temp, bytes, it, d_out, seg_count_arg(n_seg), d_off, d_off + 1, s);
    if (e) return e;
    e = cudaMallocAsync(&d_temp, bytes, s);
    if (e) return e;
    e = cub::DeviceSegmentedReduce::Sum(
        d_temp, bytes, it, d_out, seg_count_arg(n_seg), d_off, d_off + 1, s);
    cudaError_t e_free = cudaFreeAsync(d_temp, s);
    return e ? e : e_free;
}

// ---------------------------------------------------------------------------
// count — offsets difference; no CUB reduction, just a trivial kernel
// ---------------------------------------------------------------------------

inline cudaError_t seg_count(const offset_t* d_off, long long* d_out,
                             offset_t n_seg, cudaStream_t s) {
    if (n_seg <= 0) return cudaSuccess;
    const int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((n_seg + block - 1) / block);
    count_kernel<<<grid, block, 0, s>>>(d_off, d_out, n_seg);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// argmin / argmax — CUB arg-reduce into KeyValuePair, then extract the index.
// The returned key is the element's offset within its segment (segment-local),
// matching awkward's per-list argmin/argmax semantics.
// ---------------------------------------------------------------------------

template <typename T>
inline cudaError_t seg_argmin(const T* d_in, long long* d_out, offset_t n_seg,
                              const offset_t* d_off, cudaStream_t s) {
    using KVP = cub::KeyValuePair<int, T>;
    KVP* d_pairs = nullptr;
    cudaError_t e = cudaMallocAsync(reinterpret_cast<void**>(&d_pairs),
                                    sizeof(KVP) * static_cast<size_t>(n_seg), s);
    if (e) return e;

    void* d_temp = nullptr;
    size_t bytes = 0;
    e = cub::DeviceSegmentedReduce::ArgMin(
        d_temp, bytes, d_in, d_pairs, seg_count_arg(n_seg), d_off, d_off + 1, s);
    if (e) {
        cudaFreeAsync(d_pairs, s);
        return e;
    }
    e = cudaMallocAsync(&d_temp, bytes, s);
    if (e) {
        cudaFreeAsync(d_pairs, s);
        return e;
    }
    e = cub::DeviceSegmentedReduce::ArgMin(
        d_temp, bytes, d_in, d_pairs, seg_count_arg(n_seg), d_off, d_off + 1, s);

    if (!e) {
        const int block = 256;
        const unsigned int grid =
            static_cast<unsigned int>((n_seg + block - 1) / block);
        extract_keys<T><<<grid, block, 0, s>>>(d_pairs, d_out, n_seg);
        e = cudaGetLastError();
    }
    cudaFreeAsync(d_temp, s);
    cudaFreeAsync(d_pairs, s);
    return e;
}

template <typename T>
inline cudaError_t seg_argmax(const T* d_in, long long* d_out, offset_t n_seg,
                              const offset_t* d_off, cudaStream_t s) {
    using KVP = cub::KeyValuePair<int, T>;
    KVP* d_pairs = nullptr;
    cudaError_t e = cudaMallocAsync(reinterpret_cast<void**>(&d_pairs),
                                    sizeof(KVP) * static_cast<size_t>(n_seg), s);
    if (e) return e;

    void* d_temp = nullptr;
    size_t bytes = 0;
    e = cub::DeviceSegmentedReduce::ArgMax(
        d_temp, bytes, d_in, d_pairs, seg_count_arg(n_seg), d_off, d_off + 1, s);
    if (e) {
        cudaFreeAsync(d_pairs, s);
        return e;
    }
    e = cudaMallocAsync(&d_temp, bytes, s);
    if (e) {
        cudaFreeAsync(d_pairs, s);
        return e;
    }
    e = cub::DeviceSegmentedReduce::ArgMax(
        d_temp, bytes, d_in, d_pairs, seg_count_arg(n_seg), d_off, d_off + 1, s);

    if (!e) {
        const int block = 256;
        const unsigned int grid =
            static_cast<unsigned int>((n_seg + block - 1) / block);
        extract_keys<T><<<grid, block, 0, s>>>(d_pairs, d_out, n_seg);
        e = cudaGetLastError();
    }
    cudaFreeAsync(d_temp, s);
    cudaFreeAsync(d_pairs, s);
    return e;
}

}  // namespace rawkward_cccl
