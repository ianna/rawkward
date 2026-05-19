// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

// Standalone C++ argsort benchmark — no awkward-cpp headers required.
//
// Implements bench_awkward_argsort_float32 using only the C++ standard
// library.  The algorithm is identical to awkward_argsort_float32 (iota +
// std::sort / std::stable_sort with NaN-first comparators), so the timing
// comparison is apples-to-apples.

#include <algorithm>
#include <cmath>
#include <cstdint>

extern "C" {

void bench_awkward_argsort_float32(
    int64_t*       toptr,
    const float*   fromptr,
    int64_t        length,
    const int64_t* offsets,
    int64_t        offsetslength,
    bool           ascending,
    bool           stable)
{
    // Initialise with global indices.
    for (int64_t k = 0; k < length; ++k) toptr[k] = k;

    for (int64_t i = 0; i < offsetslength - 1; ++i) {
        const int64_t start = offsets[i];
        const int64_t stop  = offsets[i + 1];
        const int64_t n     = stop - start;
        if (n <= 1) continue;

        int64_t* seg = toptr + start;

        // NaN-first comparators — mirrors awkward's argsort_order_ascending /
        // argsort_order_descending template functions.
        auto cmp_asc = [&](int64_t a, int64_t b) -> bool {
            const float fa = fromptr[a], fb = fromptr[b];
            return !std::isnan(fb) && (std::isnan(fa) || fa < fb);
        };
        auto cmp_desc = [&](int64_t a, int64_t b) -> bool {
            const float fa = fromptr[a], fb = fromptr[b];
            return !std::isnan(fb) && (std::isnan(fa) || fa > fb);
        };

        if (stable) {
            ascending ? std::stable_sort(seg, seg + n, cmp_asc)
                      : std::stable_sort(seg, seg + n, cmp_desc);
        } else {
            ascending ? std::sort(seg, seg + n, cmp_asc)
                      : std::sort(seg, seg + n, cmp_desc);
        }

        // Make local (0-based within the list).
        for (int64_t j = 0; j < n; ++j) seg[j] -= start;
    }
}

} // extern "C"
