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


_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _pixmap():
    return _input_arrays(SHAPE)["pixmap"]


@pytest.mark.parametrize(
    "call,message",
    [
        (lambda: cdrizzle.invert_pixmap("not an array", np.zeros(2), None), "Invalid pixmap"),
        (lambda: cdrizzle.invert_pixmap(_pixmap(), "not an array", None), "Invalid xyout"),
        (
            lambda: cdrizzle.invert_pixmap(_pixmap(), np.zeros(2), "not an array"),
            "Invalid input bounding box",
        ),
        (lambda: cdrizzle.clip_polygon("not an array", _SQUARE), "Invalid P"),
        (lambda: cdrizzle.clip_polygon(_SQUARE, "not an array"), "Invalid Q"),
    ],
    ids=["invert_pixmap-pixmap", "invert_pixmap-xyout", "invert_pixmap-bbox", "clip-p", "clip-q"],
)
def test_invalid_arguments_raise_value_error(call, message):
    """Invalid arguments used to crash the interpreter instead of raising."""
    with pytest.raises(ValueError, match=message):
        call()
