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
    """Results written to a copy of an output array would be lost."""
    outputs = _output_arrays(SHAPE)
    outputs["output"] = modify(outputs["output"])

    with pytest.raises(ValueError, match="'output' must be a 2D"):
        _tdriz(_input_arrays(SHAPE), outputs)


@pytest.mark.parametrize("name", ["counts", "context", "outdq"])
def test_tdriz_checks_all_output_arrays(name):
    """Every output array is checked, not only 'output'."""
    outputs = _output_arrays(SHAPE)
    outputs[name] = _non_contiguous(outputs[name])

    with pytest.raises(ValueError, match=f"'{name}' must be a 2D"):
        _tdriz(_input_arrays(SHAPE), outputs)


def test_tdriz_rejects_non_contiguous_output2():
    outputs = _output_arrays(SHAPE)
    outputs["output2"] = [_non_contiguous(outputs["output2"][0])]

    with pytest.raises(ValueError, match="'output2' must be a 2D"):
        _tdriz(_input_arrays(SHAPE), outputs)


def test_tdriz_single_output2_array():
    """'output2' can be a single array instead of a list of arrays."""
    expected = _output_arrays(SHAPE)
    _tdriz(_input_arrays(SHAPE), expected)

    outputs = _output_arrays(SHAPE)
    outputs["output2"] = outputs["output2"][0]
    _tdriz(_input_arrays(SHAPE), outputs)

    np.testing.assert_array_equal(outputs["output2"], expected["output2"][0])


def test_tdriz_none_in_output2():
    outputs = _output_arrays(SHAPE)
    outputs["output2"] = [None]

    with pytest.raises(ValueError, match="Element 0 of 'output2' list is None"):
        _tdriz(_input_arrays(SHAPE), outputs)


def test_tdriz_byte_swapped_inputs():
    expected = _output_arrays(SHAPE)
    _tdriz(_input_arrays(SHAPE), expected)

    inputs = {
        name: [_byte_swapped(a) for a in arr] if isinstance(arr, list) else _byte_swapped(arr)
        for name, arr in _input_arrays(SHAPE).items()
    }
    outputs = _output_arrays(SHAPE)
    _tdriz(inputs, outputs)

    np.testing.assert_array_equal(outputs["output"], expected["output"])
    np.testing.assert_array_equal(outputs["counts"], expected["counts"])
    np.testing.assert_array_equal(outputs["context"], expected["context"])
    np.testing.assert_array_equal(outputs["output2"][0], expected["output2"][0])
    np.testing.assert_array_equal(outputs["outdq"], expected["outdq"])


@pytest.mark.parametrize("modify", [_non_contiguous, _read_only, _byte_swapped, _float64])
def test_tblot_rejects_output_not_updatable_in_place(modify):
    inputs = _input_arrays(SHAPE)
    output = modify(np.zeros(SHAPE, dtype=np.float32))

    with pytest.raises(Exception, match="'output' must be a 2D"):
        cdrizzle.tblot(inputs["input"], inputs["pixmap"], output, interp="linear")


def test_tblot_byte_swapped_inputs():
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
    with pytest.raises(ValueError, match="Invalid pixmap"):
        cdrizzle.invert_pixmap("not an array", np.zeros(2), None)


def test_invert_pixmap_invalid_xyout():
    with pytest.raises(ValueError, match="Invalid xyout"):
        cdrizzle.invert_pixmap(_pixmap(), "not an array", None)


def test_invert_pixmap_invalid_bounding_box():
    with pytest.raises(ValueError, match="Invalid input bounding box"):
        cdrizzle.invert_pixmap(_pixmap(), np.zeros(2), "not an array")


def test_clip_polygon_invalid_first_polygon():
    with pytest.raises(ValueError, match="Invalid P"):
        cdrizzle.clip_polygon("not an array", _SQUARE)


def test_clip_polygon_invalid_second_polygon():
    with pytest.raises(ValueError, match="Invalid Q"):
        cdrizzle.clip_polygon(_SQUARE, "not an array")
