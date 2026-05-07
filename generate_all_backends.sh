#!/usr/bin/env bash

set -e

LICENSE="// Copyright (c) 2026 Ianna Osborne
// SPDX-License-Identifier: BSD-3-Clause

"

TODO="// TODO: implement
"

# Full list of kernel module paths (relative to src/kernels)
MODULES=(
  "mask/bitmasked_to_bytemasked"
  "mask/bitmasked_to_indexedoption"
  "mask/bytemasked_getitem_nextcarry"
  "mask/bytemasked_numnull"
  "mask/bytemasked_overlay_mask"
  "mask/bytemasked_reduce_next"
  "mask/bytemasked_to_indexedoption"

  "index/index_nones_as_index"
  "index/indexed_fill"
  "index/indexed_flatten"
  "index/indexed_getitem_nextcarry"
  "index/indexed_index_of_nulls"
  "index/indexed_local_preparenext"
  "index/indexed_numnull"
  "index/indexed_overlay_mask"
  "index/indexed_ranges"
  "index/indexed_reduce_next"
  "index/indexed_simplify"
  "index/indexed_unique"
  "index/indexed_validity"
  "index/indexedoption_rpad_and_clip_mask_axis1"

  "list_array/broadcast_tooffsets"
  "list_array/combinations"
  "list_array/compact_offsets"
  "list_array/fill"
  "list_array/getitem_jagged"
  "list_array/getitem_next_array"
  "list_array/getitem_next_at"
  "list_array/getitem_next_range"
  "list_array/localindex"
  "list_array/min_range"
  "list_array/rpad_and_clip_length_axis1"
  "list_array/rpad_axis1"
  "list_array/validity"

  "list_offset/argsort_strings"
  "list_offset/drop_none_indexes"
  "list_offset/flatten_offsets"
  "list_offset/local_preparenext"
  "list_offset/reduce_local_nextparents"
  "list_offset/reduce_local_outoffsets"
  "list_offset/reduce_nonlocal_maxcount_offsetscopy"
  "list_offset/reduce_nonlocal_nextshifts"
  "list_offset/reduce_nonlocal_nextstarts"
  "list_offset/reduce_nonlocal_outstartsstops"
  "list_offset/reduce_nonlocal_preparenext"
  "list_offset/rpad_and_clip_axis1"
  "list_offset/rpad_axis1"
  "list_offset/rpad_length_axis1"
  "list_offset/to_regular_array"

  "regular_array/combinations"
  "regular_array/getitem_carry"
  "regular_array/getitem_jagged_expand"
  "regular_array/getitem_next_array"
  "regular_array/getitem_next_at"
  "regular_array/getitem_next_range"
  "regular_array/localindex"
  "regular_array/reduce_local_nextparents"
  "regular_array/reduce_nonlocal_preparenext"

  "record_array/reduce_nonlocal_outoffsets"

  "numpy_array/pad_zero_to_length"
  "numpy_array/prepare_utf8_to_utf32_padded"
  "numpy_array/rearrange_shifted"
  "numpy_array/reduce_adjust_starts"
  "numpy_array/reduce_mask_bytemasked"
  "numpy_array/sort_asstrings_uint8"
  "numpy_array/subrange_equal"
  "numpy_array/unique_strings_uint8"
  "numpy_array/utf8_to_utf32_padded"

  "union_array/fillindex"
  "union_array/fillna"
  "union_array/filltags"
  "union_array/flatten"
  "union_array/nestedfill_tags_index"
  "union_array/project"
  "union_array/regular_index"
  "union_array/simplify"
  "union_array/validity"

  "reduce/argmax"
  "reduce/argmin"
  "reduce/count"
  "reduce/countnonzero"
  "reduce/max"
  "reduce/min"
  "reduce/prod"
  "reduce/sum"
  "reduce/sort"
  "reduce/sorting_ranges"
  "reduce/unique_offsets"
  "reduce/unique_ranges"

  "misc/content_getitem_next_missing_jagged_getmaskstartstop"
  "misc/index_rpad_and_clip_axis0"
  "misc/index_rpad_and_clip_axis1"
  "misc/localindex"
  "misc/missing_repeat"

  "util/error"
  "util/ptr"
  "util/unicode"
)

BACKENDS=("cpu" "simd" "cuda" "hip")

echo "Generating CPU, SIMD, CUDA, and HIP kernel stubs…"

for backend in "${BACKENDS[@]}"; do
  for module in "${MODULES[@]}"; do
    dir="src/kernels/${backend}/$(dirname "$module")"
    mkdir -p "$dir"

    case "$backend" in
      cpu|simd)
        file="src/kernels/${backend}/${module}.rs"
        ;;
      cuda)
        file="src/kernels/${backend}/${module}.cu"
        ;;
      hip)
        file="src/kernels/${backend}/${module}.hip.cpp"
        ;;
    esac

    if [[ ! -f "$file" ]]; then
      echo "Creating $file"
      printf "%s%s" "$LICENSE" "$TODO" > "$file"
    else
      echo "Skipping existing file $file"
    fi
  done
done

echo "Done."

