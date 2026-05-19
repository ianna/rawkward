// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Segmented in-place sort of values using a batched bitonic network.
//
// Strategy mirrors the argsort kernels in src/kernels/hip/sort/:
//   small  (len ∈ [1,  64]):  one block (64  threads), LDS = T[64]
//   medium (len ∈ [65, 256]): one block (256 threads), LDS = T[256]
//   large  (len ∈ [257,4096]):one block (256 threads), multi-element LDS = T[4096]
//
// All three kernels are launched from the single extern "C" function; each
// self-filters by segment length and skips segments outside its tier.
//
// Output: out[i] holds the sorted value for position i (same total size as data).
// Lists longer than MAX_LARGE_N are silently left unchanged in `out`.

#include <hip/hip_runtime.h>
#include <float.h>
#include <limits>

#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        printf("HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
        return; \
    }

#define SMALL_N    64
#define MEDIUM_N  256
#define LARGE_N  4096

// -----------------------------------------------------------------------
// Helpers: numeric sentinel (padding value that sorts to the end)
// -----------------------------------------------------------------------
template <typename T> __device__ __forceinline__ T sort_sentinel();
template <> __device__ __forceinline__ float      sort_sentinel<float>()     { return FLT_MAX; }
template <> __device__ __forceinline__ double     sort_sentinel<double>()    { return DBL_MAX; }
template <> __device__ __forceinline__ int        sort_sentinel<int>()       { return INT_MAX; }
template <> __device__ __forceinline__ long long  sort_sentinel<long long>() { return LLONG_MAX; }

// -----------------------------------------------------------------------
// Tier 1 — small lists (1 .. SMALL_N elements)
// -----------------------------------------------------------------------
template <typename T>
__global__ void segmented_sort_small_kernel(
    const T* __restrict__ data,
    const long long* __restrict__ offsets,
    T* __restrict__ out,
    long long n_segments
) {
    __shared__ T s[SMALL_N];

    for (int list_id = (int)blockIdx.x;
             list_id < (int)n_segments;
             list_id += (int)gridDim.x)
    {
        const int start = (int)offsets[list_id];
        const int end   = (int)offsets[list_id + 1];
        const int len   = end - start;

        if (len == 0 || len > SMALL_N) continue;

        const int tid = (int)threadIdx.x;

        s[tid] = (tid < len) ? data[start + tid] : sort_sentinel<T>();
        __syncthreads();

        // Bitonic sort (ascending)
        for (int k = 2; k <= SMALL_N; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                const int ixj = tid ^ j;
                if (ixj > tid) {
                    const bool asc = ((tid & k) == 0);
                    if (( asc && s[tid] > s[ixj]) ||
                        (!asc && s[tid] < s[ixj])) {
                        T tmp = s[tid]; s[tid] = s[ixj]; s[ixj] = tmp;
                    }
                }
                __syncthreads();
            }
        }

        if (tid < len) {
            out[start + tid] = s[tid];
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------
// Tier 2 — medium lists (SMALL_N+1 .. MEDIUM_N elements)
// -----------------------------------------------------------------------
template <typename T>
__global__ void segmented_sort_medium_kernel(
    const T* __restrict__ data,
    const long long* __restrict__ offsets,
    T* __restrict__ out,
    long long n_segments
) {
    __shared__ T s[MEDIUM_N];

    for (int list_id = (int)blockIdx.x;
             list_id < (int)n_segments;
             list_id += (int)gridDim.x)
    {
        const int start = (int)offsets[list_id];
        const int end   = (int)offsets[list_id + 1];
        const int len   = end - start;

        if (len <= SMALL_N || len > MEDIUM_N) continue;

        const int tid = (int)threadIdx.x;

        s[tid] = (tid < len) ? data[start + tid] : sort_sentinel<T>();
        __syncthreads();

        for (int k = 2; k <= MEDIUM_N; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                const int ixj = tid ^ j;
                if (ixj > tid) {
                    const bool asc = ((tid & k) == 0);
                    if (( asc && s[tid] > s[ixj]) ||
                        (!asc && s[tid] < s[ixj])) {
                        T tmp = s[tid]; s[tid] = s[ixj]; s[ixj] = tmp;
                    }
                }
                __syncthreads();
            }
        }

        if (tid < len) {
            out[start + tid] = s[tid];
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------
// Tier 3 — large lists (MEDIUM_N+1 .. LARGE_N elements)
// -----------------------------------------------------------------------
template <typename T>
__global__ void segmented_sort_large_kernel(
    const T* __restrict__ data,
    const long long* __restrict__ offsets,
    T* __restrict__ out,
    long long n_segments
) {
    __shared__ T s[LARGE_N];

    for (int list_id = (int)blockIdx.x;
             list_id < (int)n_segments;
             list_id += (int)gridDim.x)
    {
        const int start = (int)offsets[list_id];
        const int end   = (int)offsets[list_id + 1];
        const int len   = end - start;

        if (len <= MEDIUM_N) continue;

        // Round up to next power of two; bail if beyond our LDS budget.
        int n = 1;
        while (n < len) n <<= 1;
        if (n > LARGE_N) continue;   // TODO: rocPRIM fallback

        for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
            s[i] = (i < len) ? data[start + i] : sort_sentinel<T>();
        }
        __syncthreads();

        for (int k = 2; k <= n; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
                    const int ixj = i ^ j;
                    if (ixj > i) {
                        const bool asc = ((i & k) == 0);
                        if (( asc && s[i] > s[ixj]) ||
                            (!asc && s[i] < s[ixj])) {
                            T tmp = s[i]; s[i] = s[ixj]; s[ixj] = tmp;
                        }
                    }
                }
                __syncthreads();
            }
        }

        for (int i = (int)threadIdx.x; i < len; i += (int)blockDim.x) {
            out[start + i] = s[i];
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------
// Typed launchers — fire all three tiers in sequence
// -----------------------------------------------------------------------
template <typename T>
void launch_segmented_sort(
    const T* data,
    const long long* offsets,
    T* out,
    long long n_segments,
    hipStream_t stream
) {
    if (n_segments == 0) return;

    int grid_batched = (n_segments < 65536) ? (int)n_segments : 65536;

    // Small tier (block = 64 threads)
    hipLaunchKernelGGL(
        segmented_sort_small_kernel<T>,
        dim3(grid_batched), dim3(SMALL_N), 0, stream,
        data, offsets, out, n_segments
    );
    HIP_CHECK(hipGetLastError());

    // Medium tier (block = 256 threads)
    hipLaunchKernelGGL(
        segmented_sort_medium_kernel<T>,
        dim3(grid_batched), dim3(MEDIUM_N), 0, stream,
        data, offsets, out, n_segments
    );
    HIP_CHECK(hipGetLastError());

    // Large tier (block = 256 threads, up to LARGE_N elements per list)
    hipLaunchKernelGGL(
        segmented_sort_large_kernel<T>,
        dim3(256), dim3(MEDIUM_N), 0, stream,
        data, offsets, out, n_segments
    );
    HIP_CHECK(hipGetLastError());
}

// -----------------------------------------------------------------------
// C API entry point (for Rust / Python FFI)
// -----------------------------------------------------------------------
extern "C" void awkward_hip_segmented_sort(
    const void* data,
    const long long* offsets,
    void* out,
    long long n_segments,
    int dtype_code,          // 0=float32, 1=float64, 2=int32, 3=int64
    hipStream_t stream
) {
    switch (dtype_code) {
        case 0: launch_segmented_sort<float>(
                    static_cast<const float*>(data),
                    offsets,
                    static_cast<float*>(out),
                    n_segments, stream); break;
        case 1: launch_segmented_sort<double>(
                    static_cast<const double*>(data),
                    offsets,
                    static_cast<double*>(out),
                    n_segments, stream); break;
        case 2: launch_segmented_sort<int>(
                    static_cast<const int*>(data),
                    offsets,
                    static_cast<int*>(out),
                    n_segments, stream); break;
        case 3: launch_segmented_sort<long long>(
                    static_cast<const long long*>(data),
                    offsets,
                    static_cast<long long*>(out),
                    n_segments, stream); break;
        default:
            printf("Unsupported dtype_code %d\n", dtype_code);
    }
}
