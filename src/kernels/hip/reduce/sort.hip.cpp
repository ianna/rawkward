// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Segmented sort of values using a batched bitonic network.
//
// Matches CPU sort (src/kernels/cpu/sort/sort.rs):
//   - ascending / descending (via ascending flag)
//   - NaN handling: NaN sorts FIRST ascending, LAST descending (matching ArgsortOrd)
//   - All 11 CPU dtypes: bool, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64
//   - parentslength: output elements at index >= parentslength are not written
//   - stable: accepted for API parity; bitonic sort is inherently unstable (ignored)
//
// Strategy: three-tier batched bitonic network.
//   small  (len ∈ [1,   64]): block=64,  LDS=T[64]
//   medium (len ∈ [65, 256]): block=256, LDS=T[256]
//   large  (len ∈ [257,4096]):block=256, LDS=T[4096]  (rounds up to next pow2)
//
// Segments longer than 4096 elements are passed through unchanged in `out`.
//
// dtype_code: 0=bool, 1=i8, 2=u8, 3=i16, 4=u16, 5=i32, 6=u32, 7=i64, 8=u64,
//             9=f32, 10=f64

#include <hip/hip_runtime.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>

#define SMALL_N     64
#define MEDIUM_N   256
#define LARGE_N   4096

// ── Padding sentinels (sort to end, filtered out before write-back) ──────────

template <typename T> __device__ __forceinline__ T sort_sentinel();
template <> __device__ __forceinline__ bool               sort_sentinel<bool>()               { return true; }
template <> __device__ __forceinline__ signed char        sort_sentinel<signed char>()        { return SCHAR_MAX; }
template <> __device__ __forceinline__ unsigned char      sort_sentinel<unsigned char>()      { return UCHAR_MAX; }
template <> __device__ __forceinline__ short              sort_sentinel<short>()              { return SHRT_MAX; }
template <> __device__ __forceinline__ unsigned short     sort_sentinel<unsigned short>()     { return USHRT_MAX; }
template <> __device__ __forceinline__ int                sort_sentinel<int>()                { return INT_MAX; }
template <> __device__ __forceinline__ unsigned int       sort_sentinel<unsigned int>()       { return UINT_MAX; }
template <> __device__ __forceinline__ long long          sort_sentinel<long long>()          { return LLONG_MAX; }
template <> __device__ __forceinline__ unsigned long long sort_sentinel<unsigned long long>() { return ULLONG_MAX; }
template <> __device__ __forceinline__ float              sort_sentinel<float>()              { return FLT_MAX; }
template <> __device__ __forceinline__ double             sort_sentinel<double>()             { return DBL_MAX; }

// ── NaN-aware key mapping ─────────────────────────────────────────────────────
// Ascending (NaN first):  NaN → -MAX  (smallest possible key)
// Descending (NaN last):  NaN → +MAX  (largest possible key)
// For non-float types there is no NaN; pass through unchanged.

template <typename T>
__device__ __forceinline__ T nan_key(T x, int /*ascending*/) { return x; }

template <>
__device__ __forceinline__ float nan_key<float>(float x, int ascending) {
    if (__isnanf(x)) return ascending ? -FLT_MAX : FLT_MAX;
    return x;
}
template <>
__device__ __forceinline__ double nan_key<double>(double x, int ascending) {
    if (__isnan(x)) return ascending ? -DBL_MAX : DBL_MAX;
    return x;
}

// ── Bitonic compare-and-swap ──────────────────────────────────────────────────
// asc_dir: direction of this sub-sequence (true = ascending sub-sequence).
// For ascending final output: asc_dir = ((tid & k) == 0)
// For descending final output: asc_dir = ((tid & k) != 0)  (flipped)
// NaN keys applied before comparison.

template <typename T>
__device__ __forceinline__ void bitonic_cas(
    T* s, int lo, int hi, int asc_dir, int ascending
) {
    T a = s[lo], b = s[hi];
    T ka = nan_key(a, ascending);
    T kb = nan_key(b, ascending);
    bool swap_needed = asc_dir ? (ka > kb) : (ka < kb);
    if (swap_needed) { s[lo] = b; s[hi] = a; }
}

// ── Tier 1 — small (1..SMALL_N elements) ─────────────────────────────────────

template <typename T>
__global__ void segmented_sort_small_kernel(
    const T* __restrict__ data,
    const long long* __restrict__ offsets,
    T* __restrict__ out,
    long long n_segments,
    long long parentslength,
    int ascending
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

        for (int k = 2; k <= SMALL_N; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                const int ixj = tid ^ j;
                if (ixj > tid) {
                    int asc_dir = ascending ? ((tid & k) == 0) : ((tid & k) != 0);
                    bitonic_cas(s, tid, ixj, asc_dir, ascending);
                }
                __syncthreads();
            }
        }

        if (tid < len && (long long)(start + tid) < parentslength) {
            out[start + tid] = s[tid];
        }
        __syncthreads();
    }
}

// ── Tier 2 — medium (SMALL_N+1..MEDIUM_N elements) ───────────────────────────

template <typename T>
__global__ void segmented_sort_medium_kernel(
    const T* __restrict__ data,
    const long long* __restrict__ offsets,
    T* __restrict__ out,
    long long n_segments,
    long long parentslength,
    int ascending
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
                    int asc_dir = ascending ? ((tid & k) == 0) : ((tid & k) != 0);
                    bitonic_cas(s, tid, ixj, asc_dir, ascending);
                }
                __syncthreads();
            }
        }

        if (tid < len && (long long)(start + tid) < parentslength) {
            out[start + tid] = s[tid];
        }
        __syncthreads();
    }
}

// ── Tier 3 — large (MEDIUM_N+1..LARGE_N elements) ────────────────────────────

template <typename T>
__global__ void segmented_sort_large_kernel(
    const T* __restrict__ data,
    const long long* __restrict__ offsets,
    T* __restrict__ out,
    long long n_segments,
    long long parentslength,
    int ascending
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

        int n = 1;
        while (n < len) n <<= 1;
        if (n > LARGE_N) continue;   // beyond LDS budget: pass through

        for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
            s[i] = (i < len) ? data[start + i] : sort_sentinel<T>();
        }
        __syncthreads();

        for (int k = 2; k <= n; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
                    const int ixj = i ^ j;
                    if (ixj > i) {
                        int asc_dir = ascending ? ((i & k) == 0) : ((i & k) != 0);
                        bitonic_cas(s, i, ixj, asc_dir, ascending);
                    }
                }
                __syncthreads();
            }
        }

        for (int i = (int)threadIdx.x; i < len; i += (int)blockDim.x) {
            if ((long long)(start + i) < parentslength) {
                out[start + i] = s[i];
            }
        }
        __syncthreads();
    }
}

// ── Pass-through kernel for segments > LARGE_N ───────────────────────────────
// Copies data → out unchanged for any segment that both tiers skipped.

template <typename T>
__global__ void segmented_sort_passthrough_kernel(
    const T* __restrict__ data,
    const long long* __restrict__ offsets,
    T* __restrict__ out,
    long long n_segments,
    long long parentslength
) {
    long long flat = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread is responsible for one flat position.
    // Determine which segment it falls in.
    // Only copy for segments where len > LARGE_N.
    // (Segments <= LARGE_N are handled by the three tiers above.)
    // We can't easily binary-search here, so launch one thread per element
    // of the whole flat array and check if its segment is oversized.
    // This is used only for edge-case segments, so the extra overhead is fine.

    // The kernel is launched with enough threads to cover all flat positions.
    // We find the segment by linear scan of offsets — acceptable since this
    // path only fires for segments > 4096 which are rare.
    (void)flat; (void)data; (void)offsets; (void)out; (void)n_segments; (void)parentslength;
    // NOTE: passthrough is handled inline: if the tier skips a segment,
    // the caller is responsible for pre-filling out = data before launch.
    // (see launch_segmented_sort comments)
}

// ── Typed launcher ────────────────────────────────────────────────────────────

template <typename T>
static void launch_segmented_sort(
    const T*         data,
    const long long* offsets,
    T*               out,
    long long        n_segments,
    long long        parentslength,
    int              ascending,
    hipStream_t      stream
) {
    if (n_segments == 0) return;

    int grid = (int)((n_segments < 65536) ? n_segments : 65536);

    hipLaunchKernelGGL(
        segmented_sort_small_kernel<T>,
        dim3(grid), dim3(SMALL_N), 0, stream,
        data, offsets, out, n_segments, parentslength, ascending
    );
    hipLaunchKernelGGL(
        segmented_sort_medium_kernel<T>,
        dim3(grid), dim3(MEDIUM_N), 0, stream,
        data, offsets, out, n_segments, parentslength, ascending
    );
    hipLaunchKernelGGL(
        segmented_sort_large_kernel<T>,
        dim3(256), dim3(MEDIUM_N), 0, stream,
        data, offsets, out, n_segments, parentslength, ascending
    );
}

// ── C API ─────────────────────────────────────────────────────────────────────
//
// dtype_code:  0=bool, 1=i8, 2=u8, 3=i16, 4=u16, 5=i32, 6=u32, 7=i64, 8=u64,
//              9=f32, 10=f64
//
// ascending:   1 = ascending (NaN first), 0 = descending (NaN last)
// stable:      accepted for API parity with CPU; always ignored (bitonic is unstable)
// parentslength: only out[0..parentslength) is written; caller allocates accordingly.
//               Pass parentslength == total flat length for the normal case.
//
// Caller must pre-fill out[i] = data[i] for segments > LARGE_N (4096) before
// calling this function, so those oversized segments pass through unchanged.

extern "C" void awkward_hip_segmented_sort(
    const void*      data,
    const long long* offsets,
    void*            out,
    long long        n_segments,
    long long        parentslength,
    int              dtype_code,
    int              ascending,
    int              /*stable*/,   // ignored
    hipStream_t      stream
) {
#define LAUNCH(TYPE) \
    launch_segmented_sort<TYPE>( \
        static_cast<const TYPE*>(data), offsets, \
        static_cast<TYPE*>(out), \
        n_segments, parentslength, ascending, stream)

    switch (dtype_code) {
        case  0: LAUNCH(bool);               break;
        case  1: LAUNCH(signed char);        break;
        case  2: LAUNCH(unsigned char);      break;
        case  3: LAUNCH(short);              break;
        case  4: LAUNCH(unsigned short);     break;
        case  5: LAUNCH(int);                break;
        case  6: LAUNCH(unsigned int);       break;
        case  7: LAUNCH(long long);          break;
        case  8: LAUNCH(unsigned long long); break;
        case  9: LAUNCH(float);              break;
        case 10: LAUNCH(double);             break;
        default: break;
    }
#undef LAUNCH
}
