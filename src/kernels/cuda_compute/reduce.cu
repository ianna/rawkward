// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause
//
// C ABI over the CUB segmented reducers in `cccl_common.cuh`. Each function is
// a host entry point that Rust (`reduce.rs`) calls by symbol name through an
// `extern "C"` block, passing raw device pointers and a `cudaStream_t` (the same
// underlying handle as the Driver-API `CUstream` the backend created). The
// return value is the CUDA error code (0 == cudaSuccess).
//
// build.rs compiles this TU with `nvcc -c` (whole-program device compilation,
// so CUB's device code is self-contained — no separate device link) into a
// static archive that is linked with `-lcudart`.
//
// BUILD-HOST VERIFICATION: not compilable without the CUDA toolkit/CUB; see the
// note in cccl_common.cuh.

#include "cccl_common.cuh"

using rawkward_cccl::offset_t;

extern "C" {

// ---- sum ------------------------------------------------------------------
int rawkward_cccl_sum_f32(const float* d_in, const offset_t* d_off, float* d_out,
                          offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_sum<float>(d_in, d_out, n_seg, d_off,
                                      static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_sum_f64(const double* d_in, const offset_t* d_off,
                          double* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_sum<double>(d_in, d_out, n_seg, d_off,
                                       static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_sum_i32(const int* d_in, const offset_t* d_off, int* d_out,
                          offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_sum<int>(d_in, d_out, n_seg, d_off,
                                    static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_sum_i64(const offset_t* d_in, const offset_t* d_off,
                          offset_t* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_sum<offset_t>(d_in, d_out, n_seg, d_off,
                                         static_cast<cudaStream_t>(stream)));
}

// ---- min ------------------------------------------------------------------
int rawkward_cccl_min_f32(const float* d_in, const offset_t* d_off, float* d_out,
                          offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_min<float>(d_in, d_out, n_seg, d_off,
                                      static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_min_f64(const double* d_in, const offset_t* d_off,
                          double* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_min<double>(d_in, d_out, n_seg, d_off,
                                       static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_min_i32(const int* d_in, const offset_t* d_off, int* d_out,
                          offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_min<int>(d_in, d_out, n_seg, d_off,
                                    static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_min_i64(const offset_t* d_in, const offset_t* d_off,
                          offset_t* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_min<offset_t>(d_in, d_out, n_seg, d_off,
                                         static_cast<cudaStream_t>(stream)));
}

// ---- max ------------------------------------------------------------------
int rawkward_cccl_max_f32(const float* d_in, const offset_t* d_off, float* d_out,
                          offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_max<float>(d_in, d_out, n_seg, d_off,
                                      static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_max_f64(const double* d_in, const offset_t* d_off,
                          double* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_max<double>(d_in, d_out, n_seg, d_off,
                                       static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_max_i32(const int* d_in, const offset_t* d_off, int* d_out,
                          offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_max<int>(d_in, d_out, n_seg, d_off,
                                    static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_max_i64(const offset_t* d_in, const offset_t* d_off,
                          offset_t* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_max<offset_t>(d_in, d_out, n_seg, d_off,
                                         static_cast<cudaStream_t>(stream)));
}

// ---- prod -----------------------------------------------------------------
int rawkward_cccl_prod_f32(const float* d_in, const offset_t* d_off,
                           float* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_prod<float>(d_in, d_out, n_seg, d_off,
                                       static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_prod_f64(const double* d_in, const offset_t* d_off,
                           double* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_prod<double>(d_in, d_out, n_seg, d_off,
                                        static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_prod_i32(const int* d_in, const offset_t* d_off, int* d_out,
                           offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_prod<int>(d_in, d_out, n_seg, d_off,
                                     static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_prod_i64(const offset_t* d_in, const offset_t* d_off,
                           offset_t* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(
        rawkward_cccl::seg_prod<offset_t>(d_in, d_out, n_seg, d_off,
                                          static_cast<cudaStream_t>(stream)));
}

// ---- countnonzero (output i64) --------------------------------------------
int rawkward_cccl_countnonzero_f32(const float* d_in, const offset_t* d_off,
                                   offset_t* d_out, offset_t n_seg,
                                   void* stream) {
    return static_cast<int>(rawkward_cccl::seg_countnonzero<float>(
        d_in, d_out, n_seg, d_off, static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_countnonzero_f64(const double* d_in, const offset_t* d_off,
                                   offset_t* d_out, offset_t n_seg,
                                   void* stream) {
    return static_cast<int>(rawkward_cccl::seg_countnonzero<double>(
        d_in, d_out, n_seg, d_off, static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_countnonzero_i32(const int* d_in, const offset_t* d_off,
                                   offset_t* d_out, offset_t n_seg,
                                   void* stream) {
    return static_cast<int>(rawkward_cccl::seg_countnonzero<int>(
        d_in, d_out, n_seg, d_off, static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_countnonzero_i64(const offset_t* d_in, const offset_t* d_off,
                                   offset_t* d_out, offset_t n_seg,
                                   void* stream) {
    return static_cast<int>(rawkward_cccl::seg_countnonzero<offset_t>(
        d_in, d_out, n_seg, d_off, static_cast<cudaStream_t>(stream)));
}

// ---- count (dtype-independent: offsets difference, output i64) -------------
int rawkward_cccl_count(const offset_t* d_off, offset_t* d_out, offset_t n_seg,
                        void* stream) {
    return static_cast<int>(rawkward_cccl::seg_count(
        d_off, d_out, n_seg, static_cast<cudaStream_t>(stream)));
}

// ---- argmin (output i64 segment-local index) ------------------------------
int rawkward_cccl_argmin_f32(const float* d_in, const offset_t* d_off,
                             offset_t* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(rawkward_cccl::seg_argmin<float>(
        d_in, d_out, n_seg, d_off, static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_argmin_f64(const double* d_in, const offset_t* d_off,
                             offset_t* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(rawkward_cccl::seg_argmin<double>(
        d_in, d_out, n_seg, d_off, static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_argmin_i32(const int* d_in, const offset_t* d_off,
                             offset_t* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(rawkward_cccl::seg_argmin<int>(
        d_in, d_out, n_seg, d_off, static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_argmin_i64(const offset_t* d_in, const offset_t* d_off,
                             offset_t* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(rawkward_cccl::seg_argmin<offset_t>(
        d_in, d_out, n_seg, d_off, static_cast<cudaStream_t>(stream)));
}

// ---- argmax (output i64 segment-local index) ------------------------------
int rawkward_cccl_argmax_f32(const float* d_in, const offset_t* d_off,
                             offset_t* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(rawkward_cccl::seg_argmax<float>(
        d_in, d_out, n_seg, d_off, static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_argmax_f64(const double* d_in, const offset_t* d_off,
                             offset_t* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(rawkward_cccl::seg_argmax<double>(
        d_in, d_out, n_seg, d_off, static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_argmax_i32(const int* d_in, const offset_t* d_off,
                             offset_t* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(rawkward_cccl::seg_argmax<int>(
        d_in, d_out, n_seg, d_off, static_cast<cudaStream_t>(stream)));
}
int rawkward_cccl_argmax_i64(const offset_t* d_in, const offset_t* d_off,
                             offset_t* d_out, offset_t n_seg, void* stream) {
    return static_cast<int>(rawkward_cccl::seg_argmax<offset_t>(
        d_in, d_out, n_seg, d_off, static_cast<cudaStream_t>(stream)));
}

}  // extern "C"
