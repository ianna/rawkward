// SPDX-License-Identifier: BSD-3-Clause

def awkward_reduce_argmin(
    toptr,
    fromptr,
    offsets,
    paren,
    starts,
    outlength,
):
    index_dtype = parents_data.dtype

    def segment_reduce_argmin(segment_id):
        start_idx = start_o[segment_id]
        end_idx = end_o[segment_id]
        segment = input_data[start_idx:end_idx]
        if len(segment) == 0:
            return -1
        # return a global index
        return np.argmin(segment) + start_idx

    start_o = offsets[:-1]
    end_o = offsets[1:]

    # Perform the segmented reduce
    # type_wrapper is always cp.int64
    type_wrapper = cp.dtype(index_dtype).type
    segment_ids = CountingIterator(type_wrapper(0))
    # TODO: try using segmented_reduce instead when https://github.com/NVIDIA/cccl/issues/6171 is fixed
    unary_transform(
        d_in=segment_ids, d_out=result, op=segment_reduce_argmin, num_items=outlength
    )
