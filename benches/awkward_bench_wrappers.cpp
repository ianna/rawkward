// awkward_bench_wrappers.cpp
//
// Thin extern "C" thunks around the awkward C++ kernels. The awkward
// originals return an `ERROR` struct; we discard it (we never feed
// pathological input) and expose plain void-returning symbols that Rust
// can call through FFI.
//
// Signatures here match `awkward-cpp/include/awkward/kernels.h` exactly
// — verified against the user-supplied 2026-04-26 generated header.
//
// Note: most reduce kernels take an `offsets` array (and argmax/argmin
// also take a `starts` array) that the equivalent Rust kernels don't
// need. The bench harness synthesises matching arrays so the C++ side
// gets valid input and the workload-shape stays equivalent.

#include "awkward/kernels.h"
#include <cstdint>

extern "C" {

// ── Reductions ────────────────────────────────────────────────────────────────

void bench_awkward_reduce_sum_int64_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    const int64_t* offsets,
    int64_t lenparents,
    int64_t outlength)
{
    awkward_reduce_sum_int64_int64_64(
        toptr, fromptr, parents, offsets, lenparents, outlength);
}

void bench_awkward_reduce_max_int64_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    const int64_t* offsets,
    int64_t lenparents,
    int64_t outlength,
    int64_t identity)
{
    awkward_reduce_max_int64_int64_64(
        toptr, fromptr, parents, offsets, lenparents, outlength, identity);
}

void bench_awkward_reduce_min_int64_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    const int64_t* offsets,
    int64_t lenparents,
    int64_t outlength,
    int64_t identity)
{
    awkward_reduce_min_int64_int64_64(
        toptr, fromptr, parents, offsets, lenparents, outlength, identity);
}

void bench_awkward_reduce_prod_int64_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    const int64_t* offsets,
    int64_t lenparents,
    int64_t outlength)
{
    awkward_reduce_prod_int64_int64_64(
        toptr, fromptr, parents, offsets, lenparents, outlength);
}

void bench_awkward_reduce_argmax_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    const int64_t* offsets,
    int64_t lenparents,
    const int64_t* starts,
    int64_t outlength)
{
    awkward_reduce_argmax_int64_64(
        toptr, fromptr, parents, offsets, lenparents, starts, outlength);
}

void bench_awkward_reduce_argmin_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    const int64_t* offsets,
    int64_t lenparents,
    const int64_t* starts,
    int64_t outlength)
{
    awkward_reduce_argmin_int64_64(
        toptr, fromptr, parents, offsets, lenparents, starts, outlength);
}

void bench_awkward_reduce_count_64(
    int64_t* toptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength)
{
    awkward_reduce_count_64(toptr, parents, lenparents, outlength);
}

void bench_awkward_reduce_countnonzero_int64_64(
    int64_t* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    int64_t lenparents,
    int64_t outlength)
{
    awkward_reduce_countnonzero_int64_64(
        toptr, fromptr, parents, lenparents, outlength);
}

void bench_awkward_reduce_sum_bool_int64_64(
    bool* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    const int64_t* offsets,
    int64_t lenparents,
    int64_t outlength)
{
    awkward_reduce_sum_bool_int64_64(
        toptr, fromptr, parents, offsets, lenparents, outlength);
}

void bench_awkward_reduce_prod_bool_int64_64(
    bool* toptr,
    const int64_t* fromptr,
    const int64_t* parents,
    const int64_t* offsets,
    int64_t lenparents,
    int64_t outlength)
{
    awkward_reduce_prod_bool_int64_64(
        toptr, fromptr, parents, offsets, lenparents, outlength);
}

// ── Per-layout helpers ────────────────────────────────────────────────────────

void bench_awkward_ListArray64_compact_offsets_64(
    int64_t* tooffsets,
    const int64_t* fromstarts,
    const int64_t* fromstops,
    int64_t length)
{
    awkward_ListArray64_compact_offsets_64(
        tooffsets, fromstarts, fromstops, length);
}

void bench_awkward_missing_repeat_64(
    int64_t* outindex,
    const int64_t* index,
    int64_t indexlength,
    int64_t repetitions,
    int64_t regularsize)
{
    awkward_missing_repeat_64(
        outindex, index, indexlength, repetitions, regularsize);
}

} // extern "C"
// Note: bench_awkward_argsort_float32 is defined in argsort_cxx_impl.cpp
// (standalone, no awkward headers required) rather than here.
