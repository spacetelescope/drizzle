import numpy as np
import pytest

from drizzle import cdrizzle

SHAPE = (10, 12)


def test_cdrizzle():
    """
    Call C unit tests for cdrizzle, which are in the src/tests directory
    """

    size = 100
    data = np.zeros((size, size), dtype="float32")
    weights = np.ones((size, size), dtype="float32")

    pixmap = np.indices((size, size), dtype="float64")
    pixmap = pixmap.transpose()

    output_data = np.zeros((size, size), dtype="float32")
    output_counts = np.zeros((size, size), dtype="float32")
    output_context = np.zeros((size, size), dtype="int32")

    cdrizzle.test_cdrizzle(
        data,
        weights,
        pixmap,
        output_data,
        output_counts,
        output_context,
    )


def _input_arrays(shape):
    y, x = np.indices(shape)
    data = (x + 10.0 * y).astype(np.float32)
    # shift by a fraction of a pixel so that output pixels mix input pixels
    pixmap = np.dstack([x, y]).astype(np.float64) + 0.25
    return {
        "input": data,
        "weights": np.ones(shape, dtype=np.float32),
        "pixmap": pixmap,
        "input2": [2.0 * data],
        "dq": np.ones(shape, dtype=np.uint32),
    }


def _output_arrays(shape):
    return {
        "output": np.zeros(shape, dtype=np.float32),
        "counts": np.zeros(shape, dtype=np.float32),
        "context": np.zeros(shape, dtype=np.int32),
        "output2": [np.zeros(shape, dtype=np.float32)],
        "outdq": np.zeros(shape, dtype=np.uint32),
    }


def _tdriz(inputs, outputs):
    cdrizzle.tdriz(
        **inputs,
        **outputs,
        uniqid=1,
        pixfrac=1.0,
        kernel="square",
        in_units="cps",
        expscale=1.0,
        wtscale=1.0,
        fillstr="INDEF",
    )


def _non_contiguous(arr):
    return np.repeat(arr, 2, axis=1)[:, ::2]


def _read_only(arr):
    arr = arr.copy()
    arr.flags.writeable = False
    return arr


def _byte_swapped(arr):
    return arr.astype(arr.dtype.newbyteorder())


def _float64(arr):
    return arr.astype(np.float64)


@pytest.mark.parametrize("modify", [_non_contiguous, _read_only, _byte_swapped, _float64])
def test_tdriz_rejects_output_not_updatable_in_place(modify):
    """
    tdriz raises ValueError for an 'output' array that it cannot update in
    place: non-contiguous, read-only, byte-swapped, or float64. It used to
    write the results to a temporary copy, and they were lost.
    """
    outputs = _output_arrays(SHAPE)
    outputs["output"] = modify(outputs["output"])

    with pytest.raises(ValueError, match="'output' must be a 2D"):
        _tdriz(_input_arrays(SHAPE), outputs)


@pytest.mark.parametrize("name", ["counts", "context", "outdq"])
def test_tdriz_checks_all_output_arrays(name):
    """
    tdriz also raises ValueError for non-contiguous 'counts', 'context', and
    'outdq' arrays, not only for 'output'.
    """
    outputs = _output_arrays(SHAPE)
    outputs[name] = _non_contiguous(outputs[name])

    with pytest.raises(ValueError, match=f"'{name}' must be a 2D"):
        _tdriz(_input_arrays(SHAPE), outputs)


def test_tdriz_rejects_non_contiguous_output2():
    """tdriz raises ValueError for a non-contiguous array in the 'output2' list."""
    outputs = _output_arrays(SHAPE)
    outputs["output2"] = [_non_contiguous(outputs["output2"][0])]

    with pytest.raises(ValueError, match="'output2' must be a 2D"):
        _tdriz(_input_arrays(SHAPE), outputs)


def test_tdriz_single_output2_array():
    """
    tdriz gives the same result when 'output2' is a single array as when it
    is a list holding that array. A single array used to crash tdriz.
    """
    expected = _output_arrays(SHAPE)
    _tdriz(_input_arrays(SHAPE), expected)

    outputs = _output_arrays(SHAPE)
    outputs["output2"] = outputs["output2"][0]
    _tdriz(_input_arrays(SHAPE), outputs)

    np.testing.assert_array_equal(outputs["output2"], expected["output2"][0])


def test_tdriz_none_in_output2():
    """
    tdriz raises ValueError when the 'output2' list contains None. This used
    to crash tdriz.
    """
    outputs = _output_arrays(SHAPE)
    outputs["output2"] = [None]

    with pytest.raises(ValueError, match="Element 0 of 'output2' list is None"):
        _tdriz(_input_arrays(SHAPE), outputs)


# Input arrays in non-native byte order, as read from FITS files, used to be
# misread by tdriz.


def test_tdriz_byte_swapped_data():
    """
    tdriz gives the same 'output' when the 'input' data are in non-native
    byte order as when they are in native byte order.
    """
    inputs = _input_arrays(SHAPE)
    expected = _output_arrays(SHAPE)
    _tdriz(inputs, expected)

    inputs["input"] = _byte_swapped(inputs["input"])
    outputs = _output_arrays(SHAPE)
    _tdriz(inputs, outputs)

    np.testing.assert_array_equal(outputs["output"], expected["output"])


def test_tdriz_byte_swapped_weights():
    """
    tdriz gives the same 'counts' when the weights are in non-native byte
    order as when they are in native byte order.
    """
    inputs = _input_arrays(SHAPE)
    expected = _output_arrays(SHAPE)
    _tdriz(inputs, expected)

    inputs["weights"] = _byte_swapped(inputs["weights"])
    outputs = _output_arrays(SHAPE)
    _tdriz(inputs, outputs)

    np.testing.assert_array_equal(outputs["counts"], expected["counts"])


def test_tdriz_byte_swapped_pixmap():
    """
    tdriz gives the same 'output' and 'counts' when the pixel map is in
    non-native byte order as when it is in native byte order.
    """
    inputs = _input_arrays(SHAPE)
    expected = _output_arrays(SHAPE)
    _tdriz(inputs, expected)

    inputs["pixmap"] = _byte_swapped(inputs["pixmap"])
    outputs = _output_arrays(SHAPE)
    _tdriz(inputs, outputs)

    np.testing.assert_array_equal(outputs["output"], expected["output"])
    np.testing.assert_array_equal(outputs["counts"], expected["counts"])


def test_tdriz_byte_swapped_input2():
    """
    tdriz gives the same 'output2' when the 'input2' data are in non-native
    byte order as when they are in native byte order.
    """
    inputs = _input_arrays(SHAPE)
    expected = _output_arrays(SHAPE)
    _tdriz(inputs, expected)

    inputs["input2"] = [_byte_swapped(inputs["input2"][0])]
    outputs = _output_arrays(SHAPE)
    _tdriz(inputs, outputs)

    np.testing.assert_array_equal(outputs["output2"][0], expected["output2"][0])


def test_tdriz_byte_swapped_dq():
    """
    tdriz gives the same 'outdq' when the 'dq' array is in non-native byte
    order as when it is in native byte order.
    """
    inputs = _input_arrays(SHAPE)
    expected = _output_arrays(SHAPE)
    _tdriz(inputs, expected)

    inputs["dq"] = _byte_swapped(inputs["dq"])
    outputs = _output_arrays(SHAPE)
    _tdriz(inputs, outputs)

    np.testing.assert_array_equal(outputs["outdq"], expected["outdq"])


@pytest.mark.parametrize("modify", [_non_contiguous, _read_only, _byte_swapped, _float64])
def test_tblot_rejects_output_not_updatable_in_place(modify):
    """
    tblot raises an error for an output array that it cannot update in
    place: non-contiguous, read-only, byte-swapped, or float64.
    """
    inputs = _input_arrays(SHAPE)
    output = modify(np.zeros(SHAPE, dtype=np.float32))

    with pytest.raises(Exception, match="'output' must be a 2D"):
        cdrizzle.tblot(inputs["input"], inputs["pixmap"], output, interp="linear")


def test_tblot_byte_swapped_inputs():
    """
    tblot gives the same result when the data and pixel map are in
    non-native byte order as when they are in native byte order.
    """
    inputs = _input_arrays(SHAPE)
    expected = np.zeros(SHAPE, dtype=np.float32)
    cdrizzle.tblot(inputs["input"], inputs["pixmap"], expected, interp="linear")

    output = np.zeros(SHAPE, dtype=np.float32)
    cdrizzle.tblot(
        _byte_swapped(inputs["input"]), _byte_swapped(inputs["pixmap"]), output, interp="linear"
    )

    np.testing.assert_array_equal(output, expected)


_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _pixmap():
    return _input_arrays(SHAPE)["pixmap"]


# Invalid arguments to invert_pixmap and clip_polygon used to crash the
# interpreter instead of raising an exception.


def test_invert_pixmap_invalid_pixmap():
    """invert_pixmap raises ValueError when 'pixmap' is not an array."""
    with pytest.raises(ValueError, match="Invalid pixmap"):
        cdrizzle.invert_pixmap("not an array", np.zeros(2), None)


def test_invert_pixmap_invalid_xyout():
    """invert_pixmap raises ValueError when 'xyout' is not an array."""
    with pytest.raises(ValueError, match="Invalid xyout"):
        cdrizzle.invert_pixmap(_pixmap(), "not an array", None)


def test_invert_pixmap_invalid_bounding_box():
    """invert_pixmap raises ValueError when the bounding box is not an array."""
    with pytest.raises(ValueError, match="Invalid input bounding box"):
        cdrizzle.invert_pixmap(_pixmap(), np.zeros(2), "not an array")


def test_clip_polygon_invalid_first_polygon():
    """clip_polygon raises ValueError when the first polygon is not an array."""
    with pytest.raises(ValueError, match="Invalid P"):
        cdrizzle.clip_polygon("not an array", _SQUARE)


def test_clip_polygon_invalid_second_polygon():
    """clip_polygon raises ValueError when the second polygon is not an array."""
    with pytest.raises(ValueError, match="Invalid Q"):
        cdrizzle.clip_polygon(_SQUARE, "not an array")
