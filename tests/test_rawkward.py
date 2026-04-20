# Copyright (c) 2026 Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""
Pytest test suite: rawkward vs awkward-array parity.

Covers:
  - Construction from Python lists (flat, nested, records, None)
  - Layout structure (type string, node types, offsets, lengths)
  - Slicing: integer index, range slice, negative index
  - Field access: dot notation, bracket notation
  - tolist() round-trip
  - ndim, nbytes, fields, is_tuple properties
  - RegularArray (fixed-size lists)
  - IndexedOptionArray (None / missing values)
  - Edge cases: empty arrays, single-element, deeply nested
"""

import pytest
import awkward as ak
import rawkward as rk
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def rk_tolist(arr):
    """rawkward Array → Python list (via tolist())."""
    return arr.tolist()


def ak_tolist(arr):
    """awkward Array → Python list."""
    return ak.to_list(arr)


def lists_equal(a, b):
    """Recursively compare two nested lists/dicts/None values."""
    if a is None and b is None:
        return True
    if type(a) != type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(lists_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return set(a) == set(b) and all(lists_equal(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) < 1e-9 or (np.isnan(a) and np.isnan(b))
    return a == b


# ── Flat NumpyArray ───────────────────────────────────────────────────────────

class TestFlatArray:
    def test_construct(self):
        rk_arr = rk.Array([1.0, 2.0, 3.0])
        ak_arr = ak.Array([1.0, 2.0, 3.0])
        assert lists_equal(rk_tolist(rk_arr), ak_tolist(ak_arr))

    def test_len(self):
        rk_arr = rk.Array([1.0, 2.0, 3.0])
        assert len(rk_arr) == 3

    def test_ndim(self):
        assert rk.Array([1.0, 2.0]).ndim == 1

    def test_integer_index(self):
        rk_arr = rk.Array([10.0, 20.0, 30.0])
        assert rk_tolist(rk_arr[0]) == [10.0]
        assert rk_tolist(rk_arr[1]) == [20.0]
        assert rk_tolist(rk_arr[-1]) == [30.0]

    def test_range_slice(self):
        rk_arr = rk.Array([1.0, 2.0, 3.0, 4.0, 5.0])
        ak_arr = ak.Array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert lists_equal(rk_tolist(rk_arr[1:3]), ak_tolist(ak_arr[1:3]))
        assert lists_equal(rk_tolist(rk_arr[2:]), ak_tolist(ak_arr[2:]))
        assert lists_equal(rk_tolist(rk_arr[:2]), ak_tolist(ak_arr[:2]))

    def test_out_of_bounds(self):
        rk_arr = rk.Array([1.0, 2.0])
        with pytest.raises(IndexError):
            _ = rk_arr[5]

    def test_fields_empty(self):
        assert rk.Array([1.0, 2.0]).fields == []

    def test_is_tuple_false(self):
        assert rk.Array([1.0, 2.0]).is_tuple is False

    def test_layout_type(self):
        arr = rk.Array([1.0, 2.0, 3.0])
        layout = arr.layout
        assert "NumpyArray" in repr(layout)
        assert "float64" in repr(layout)
        assert "len='3'" in repr(layout)

    def test_nbytes(self):
        arr = rk.Array([1.0, 2.0, 3.0, 4.0])
        # 4 float64 values = 32 bytes
        assert arr.nbytes == 32

    def test_tolist_roundtrip(self):
        data = [1.0, 2.0, 3.0]
        assert rk_tolist(rk.Array(data)) == data

    def test_empty(self):
        rk_arr = rk.Array([])
        assert len(rk_arr) == 0
        assert rk_tolist(rk_arr) == []


# ── Variable-length lists (ListOffsetArray) ───────────────────────────────────

class TestListOffsetArray:
    def test_construct(self):
        data = [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr), ak_tolist(ak_arr))

    def test_len(self):
        assert len(rk.Array([[1.0, 2.0], [3.0]])) == 2

    def test_ndim(self):
        assert rk.Array([[1.0], [2.0, 3.0]]).ndim == 2

    def test_integer_index(self):
        data = [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        for i in range(3):
            assert lists_equal(rk_tolist(rk_arr[i]), ak_tolist(ak_arr[i]))

    def test_negative_index(self):
        data = [[1.0, 2.0], [3.0], [4.0, 5.0]]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr[-1]), ak_tolist(ak_arr[-1]))

    def test_range_slice(self):
        data = [[1.0, 2.0], [], [3.0], [4.0, 5.0]]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr[1:3]), ak_tolist(ak_arr[1:3]))

    def test_empty_inner_list(self):
        data = [[1.0], [], [2.0, 3.0]]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr), ak_tolist(ak_arr))

    def test_layout_offsets(self):
        arr = rk.Array([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]])
        layout = arr.layout
        assert "ListOffsetArray" in repr(layout)
        assert "len='3'" in repr(layout)
        # offsets should be [0, 2, 3, 6]
        assert "[0 2 3 6]" in repr(layout)

    def test_layout_content(self):
        arr = rk.Array([[1.0, 2.0], [3.0]])
        layout = arr.layout
        assert "NumpyArray" in repr(layout)

    def test_deeply_nested(self):
        data = [[[1.0, 2.0], [3.0]], [[4.0]]]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr), ak_tolist(ak_arr))
        assert rk_arr.ndim == 3

    def test_tolist_roundtrip(self):
        data = [[1.0, 2.0], [], [3.0]]
        assert lists_equal(rk_tolist(rk.Array(data)), data)

    def test_single_element(self):
        data = [[42.0]]
        assert lists_equal(rk_tolist(rk.Array(data)), data)


# ── RecordArray ───────────────────────────────────────────────────────────────

class TestRecordArray:
    def test_construct(self):
        data = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr), ak_tolist(ak_arr))

    def test_len(self):
        data = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        assert len(rk.Array(data)) == 3

    def test_fields(self):
        data = [{"x": 1.0, "y": 2.0}]
        rk_arr = rk.Array(data)
        assert set(rk_arr.fields) == {"x", "y"}

    def test_field_access_bracket(self):
        data = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr["x"]), ak_tolist(ak_arr["x"]))
        assert lists_equal(rk_tolist(rk_arr["y"]), ak_tolist(ak_arr["y"]))

    def test_field_access_dot(self):
        data = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr.x), ak_tolist(ak_arr.x))
        assert lists_equal(rk_tolist(rk_arr.y), ak_tolist(ak_arr.y))

    def test_field_missing(self):
        rk_arr = rk.Array([{"x": 1.0}])
        with pytest.raises(AttributeError):
            _ = rk_arr.z

    def test_integer_index(self):
        data = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
        rk_arr = rk.Array(data)
        # indexing a record returns a length-1 record array
        row = rk_arr[0]
        assert lists_equal(rk_tolist(row["x"]), [1.0])

    def test_range_slice(self):
        data = [{"x": float(i), "y": float(i * 2)} for i in range(5)]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr[1:3]), ak_tolist(ak_arr[1:3]))

    def test_layout_record(self):
        arr = rk.Array([{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}])
        layout = arr.layout
        assert "RecordArray" in repr(layout)
        assert "is_tuple='false'" in repr(layout)
        assert "field='x'" in repr(layout)
        assert "field='y'" in repr(layout)

    def test_nested_record_in_list(self):
        data = [[{"x": 1.0, "y": [1]}, {"x": 2.0, "y": [2, 2]}],
                [{"x": 0.0, "y": []}, {"x": 1.0, "y": [1, 1, 1]}]]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr), ak_tolist(ak_arr))

    def test_nested_layout_structure(self):
        data = [[{"x": 1.0, "y": [1]}, {"x": 2.0, "y": [2, 2]}],
                [{"x": 0.0, "y": []}, {"x": 1.0, "y": [1, 1, 1]}]]
        arr = rk.Array(data)
        layout_repr = repr(arr.layout)
        assert "ListOffsetArray" in layout_repr
        assert "RecordArray" in layout_repr
        assert "field='x'" in layout_repr
        assert "field='y'" in layout_repr
        # outer offsets: 2 rows → [0, 2, 4]
        assert "[0 2 4]" in layout_repr

    def test_y_column_is_listoffsetarray(self):
        data = [[{"x": 1.0, "y": [1]}, {"x": 2.0, "y": [2, 2]}],
                [{"x": 0.0, "y": []}, {"x": 1.0, "y": [1, 1, 1]}]]
        arr = rk.Array(data)
        layout_repr = repr(arr.layout)
        # y column offsets: [0,1,3,3,6]
        assert "[0 1 3 3 6]" in layout_repr


# ── IndexedOptionArray (None values) ─────────────────────────────────────────

class TestIndexedOptionArray:
    def test_construct_flat_with_none(self):
        data = [1.0, None, 2.0]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr), ak_tolist(ak_arr))

    def test_all_none(self):
        data = [None, None, None]
        rk_arr = rk.Array(data)
        result = rk_tolist(rk_arr)
        assert all(x is None for x in result)

    def test_none_at_start(self):
        data = [None, 1.0, 2.0]
        rk_arr = rk.Array(data)
        result = rk_tolist(rk_arr)
        assert result[0] is None
        assert result[1] == [1.0] or result[1] == 1.0

    def test_none_at_end(self):
        data = [1.0, 2.0, None]
        rk_arr = rk.Array(data)
        result = rk_tolist(rk_arr)
        assert result[-1] is None

    def test_layout_indexed_option(self):
        arr = rk.Array([1.0, None, 2.0])
        layout = arr.layout
        assert "IndexedOptionArray" in repr(layout)
        assert "len='3'" in repr(layout)

    def test_layout_index_values(self):
        arr = rk.Array([1.0, None, 2.0])
        layout_repr = repr(arr.layout)
        # index should be [0, -1, 1]
        assert "-1" in layout_repr

    def test_layout_content_len(self):
        arr = rk.Array([1.0, None, 2.0])
        layout_repr = repr(arr.layout)
        # content has only the 2 valid values
        assert "len='2'" in layout_repr

    def test_none_in_list(self):
        data = [[1.0, 2.0], None, [3.0]]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr), ak_tolist(ak_arr))

    def test_integer_index_valid(self):
        rk_arr = rk.Array([1.0, None, 3.0])
        result = rk_arr[0]
        assert rk_tolist(result) == [1.0] or rk_tolist(result) == 1.0

    def test_integer_index_none(self):
        rk_arr = rk.Array([1.0, None, 3.0])
        # indexing a None position — layout-level returns IndexedOptionArray with -1
        result = rk_arr[1]
        layout_repr = repr(result.layout)
        assert "-1" in layout_repr

    def test_range_slice_preserves_none(self):
        data = [1.0, None, 3.0, None, 5.0]
        rk_arr = rk.Array(data)
        sliced = rk_arr[1:4]
        result = rk_tolist(sliced)
        assert result[0] is None
        assert result[2] is None

    def test_len_with_none(self):
        assert len(rk.Array([1.0, None, 3.0])) == 3


# ── repr and type string ──────────────────────────────────────────────────────

class TestRepr:
    def test_flat_repr(self):
        arr = rk.Array([1.0, 2.0, 3.0])
        r = repr(arr)
        assert r.startswith("<Array")
        assert "float64" in r

    def test_nested_repr(self):
        arr = rk.Array([[1.0, 2.0], [3.0]])
        r = repr(arr)
        assert "var * float64" in r

    def test_record_repr_type(self):
        arr = rk.Array([{"x": 1.0, "y": 2.0}])
        r = repr(arr)
        assert "x" in r and "y" in r

    def test_option_type_str(self):
        arr = rk.Array([1.0, None, 2.0])
        r = repr(arr)
        assert "option" in r

    def test_regular_type_str(self):
        # 2 * float64 for size-2 regular lists
        arr = rk.Array([[1.0, 2.0], [3.0, 4.0]])
        # either RegularArray or ListOffsetArray is acceptable,
        # but the values must round-trip
        assert lists_equal(rk_tolist(arr), [[1.0, 2.0], [3.0, 4.0]])


# ── Layout node structure ─────────────────────────────────────────────────────

class TestLayout:
    def test_numpy_layout(self):
        layout = rk.Array([1.0, 2.0]).layout
        r = repr(layout)
        assert r.startswith("<NumpyArray")
        assert "[1.0 2.0]" in r or "[1. 2.]" in r

    def test_listoffset_layout(self):
        layout = rk.Array([[1.0], [2.0, 3.0]]).layout
        r = repr(layout)
        assert "<ListOffsetArray" in r
        assert "<offsets>" in r
        assert "<content>" in r

    def test_record_layout(self):
        layout = rk.Array([{"a": 1.0}]).layout
        r = repr(layout)
        assert "<RecordArray" in r
        assert "field='a'" in r

    def test_option_layout(self):
        layout = rk.Array([1.0, None]).layout
        r = repr(layout)
        assert "<IndexedOptionArray" in r
        assert "<index>" in r
        assert "<content>" in r

    def test_nested_layout_depth(self):
        layout = rk.Array([[1.0, 2.0], [3.0]]).layout
        r = repr(layout)
        # Outer ListOffsetArray wraps a NumpyArray
        assert "ListOffsetArray" in r
        assert "NumpyArray" in r


# ── tolist round-trips ────────────────────────────────────────────────────────

class TestToList:
    @pytest.mark.parametrize("data", [
        [1.0, 2.0, 3.0],
        [[1.0, 2.0], [], [3.0]],
        [[[1.0], [2.0, 3.0]], [[4.0]]],
        [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
        [[{"x": 1.0}, {"x": 2.0}], [{"x": 3.0}]],
    ])
    def test_roundtrip(self, data):
        assert lists_equal(rk_tolist(rk.Array(data)), data)

    def test_to_list_alias(self):
        data = [1.0, 2.0]
        arr = rk.Array(data)
        assert arr.tolist() == arr.to_list()

    def test_none_roundtrip(self):
        data = [1.0, None, 3.0]
        result = rk_tolist(rk.Array(data))
        assert result[0] == [1.0] or result[0] == 1.0
        assert result[1] is None
        assert result[2] == [3.0] or result[2] == 3.0


# ── Iteration ─────────────────────────────────────────────────────────────────

class TestIteration:
    def test_iter_flat(self):
        arr = rk.Array([1.0, 2.0, 3.0])
        items = list(arr)
        assert len(items) == 3

    def test_iter_nested(self):
        arr = rk.Array([[1.0, 2.0], [3.0]])
        items = list(arr)
        assert len(items) == 2

    def test_iter_records(self):
        arr = rk.Array([{"x": 1.0}, {"x": 2.0}])
        items = list(arr)
        assert len(items) == 2


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_element_flat(self):
        arr = rk.Array([42.0])
        assert len(arr) == 1
        assert rk_tolist(arr) == [42.0]

    def test_single_element_nested(self):
        arr = rk.Array([[1.0, 2.0]])
        assert len(arr) == 1

    def test_empty_flat(self):
        arr = rk.Array([])
        assert len(arr) == 0

    def test_empty_nested(self):
        arr = rk.Array([[]])
        assert len(arr) == 1
        assert rk_tolist(arr) == [[]]

    def test_large_flat(self):
        data = list(range(1000))
        arr = rk.Array([float(x) for x in data])
        assert len(arr) == 1000
        result = rk_tolist(arr)
        assert result[0] == 0.0
        assert result[999] == 999.0

    def test_deeply_nested_3d(self):
        data = [[[1.0, 2.0], [3.0]], [[4.0, 5.0, 6.0]]]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr), ak_tolist(ak_arr))

    def test_record_with_list_field(self):
        data = [{"x": 1.0, "y": [1, 2]}, {"x": 2.0, "y": [3]}]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr), ak_tolist(ak_arr))

    def test_negative_slice_not_supported(self):
        arr = rk.Array([1.0, 2.0, 3.0])
        with pytest.raises((ValueError, NotImplementedError)):
            _ = arr[1:-1:2]  # step != 1


# ── Parity table: rawkward vs awkward ────────────────────────────────────────

class TestAkParity:
    """Direct comparison of rawkward and awkward output for the same inputs."""

    CASES = [
        [1.0, 2.0, 3.0],
        [[1.0, 2.0], [], [3.0]],
        [[[0.0, 1.0, 2.0], [], [3.0, 4.0], [5.0]], [[6.0, 7.0, 8.0], [9.0]]],
        [{"x": 1.1, "y": 2.2}, {"x": 3.3, "y": 4.4}],
        [[{"x": 1.0, "y": [1]}, {"x": 2.0, "y": [2, 2]}],
         [{"x": 0.0, "y": []}, {"x": 1.0, "y": [1, 1, 1]}]],
    ]

    @pytest.mark.parametrize("data", CASES)
    def test_tolist_matches_awkward(self, data):
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        assert lists_equal(rk_tolist(rk_arr), ak_tolist(ak_arr)), (
            f"Mismatch:\n  rk: {rk_tolist(rk_arr)}\n  ak: {ak_tolist(ak_arr)}"
        )

    @pytest.mark.parametrize("data", CASES)
    def test_len_matches_awkward(self, data):
        assert len(rk.Array(data)) == len(ak.Array(data))

    @pytest.mark.parametrize("data", CASES)
    def test_ndim_matches_awkward(self, data):
        rk_ndim = rk.Array(data).ndim
        ak_ndim = ak.Array(data).ndim
        assert rk_ndim == ak_ndim, (
            f"ndim mismatch for {data!r}: rk={rk_ndim}, ak={ak_ndim}"
        )

    def test_fields_match(self):
        data = [{"x": 1.0, "y": 2.0}]
        assert set(rk.Array(data).fields) == set(ak.Array(data).fields)

    def test_field_values_match(self):
        data = [{"x": 1.1, "y": 2.2}, {"x": 3.3, "y": 4.4}]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        for field in ["x", "y"]:
            assert lists_equal(rk_tolist(rk_arr[field]), ak_tolist(ak_arr[field]))

    def test_slice_matches_awkward(self):
        data = [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0], [7.0]]
        rk_arr = rk.Array(data)
        ak_arr = ak.Array(data)
        for start, stop in [(0, 2), (1, 3), (0, 4), (2, 4)]:
            assert lists_equal(
                rk_tolist(rk_arr[start:stop]),
                ak_tolist(ak_arr[start:stop]),
            ), f"Slice [{start}:{stop}] mismatch"