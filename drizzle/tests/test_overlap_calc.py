import os
import pytest
from math import sqrt

import numpy as np

from drizzle.cdrizzle import clip_polygon, invert_pixmap


TEST_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TEST_DIR, 'data')
SQ2 = 1.0 / sqrt(2.0)


def _is_poly_eq(p1, p2, rtol=0, atol=1e-12):
    if len(p1) != len(p2):
        return False

    p1 = p1[:]
    for _ in p1:
        p1.append(p1.pop(0))
        if np.allclose(p1, p2, rtol=rtol, atol=atol):
            return True
    return False


def _coord_mapping(xin, yin):
    crpix = (289, 348)  # center of distortions
    shift = (1000, 1000)
    rmat = 2.0 * np.array([[0.78103169, 0.66712321], [-0.63246699, 0.74091539]])
    x = xin - crpix[0]
    y = yin - crpix[1]

    # add non-linear distortions
    x += 2.4e-6 * x**2 - 1.0e-7 * x * y + 3.1e-6 * y**2
    y += 1.2e-6 * x**2 - 2.0e-7 * x * y + 1.1e-6 * y**2

    x, y = np.dot(rmat, [x, y])
    x += shift[0]
    y += shift[1]

    return x, y


def _roll_vertices(polygon, n=1):
    n = n % len(polygon)
    return polygon[n:] + polygon[:n]


def test_invert_pixmap():
    yin, xin = np.indices((1000, 1200), dtype=float)
    xin = xin.flatten()
    yin = yin.flatten()

    xout, yout = _coord_mapping(xin, yin)
    xout = xout.reshape((1000, 1200))
    yout = yout.reshape((1000, 1200))
    pixmap = np.dstack([xout, yout])

    test_coords = [
        (300, 600),
        (0, 0),
        (1199, 999),
        (0, 999),
        (1199, 0),
        (200, 0),
        (0, 438),
        (1199, 432),
    ]

    for xr, yr in test_coords:
        xout_t, yout_t = _coord_mapping(xr, yr)
        xyin = invert_pixmap(pixmap, [xout_t, yout_t], [[-0.5, 1199.5], [-0.5, 999.5]])
        assert np.allclose(xyin, [xr, yr], atol=0.05)


def test_poly_intersection_with_self():
    p = [(0, 0), (1, 0), (1, 1), (0, 1)]

    for k in range(4):
        q = _roll_vertices(p, k)

        pq = clip_polygon(q, p)
        assert _is_poly_eq(pq, q)


@pytest.mark.parametrize(
    'shift', [(0.25, 0.1), (-0.25, -0.1), (-0.25, 0.1), (0.25, -0.1)],
)
def test_poly_intersection_shifted(shift):
    p = [(0, 0), (1, 0), (1, 1), (0, 1)]
    sx, sy = shift
    pq_ref = sorted(
        [
            (max(0, sx), max(0, sy)),
            (min(1, sx + 1), max(0, sy)),
            (min(1, sx + 1), min(1, sy + 1)),
            (max(0, sx), min(1, sy + 1)),
        ],
    )

    for k in range(4):
        q = [(x + sx, y + sy) for x, y in p]
        q = _roll_vertices(q, k)
        pq = clip_polygon(q, p)
        assert np.allclose(sorted(pq), pq_ref)


@pytest.mark.parametrize(
    'shift', [(0, 70), (70, 0), (0, -70), (-70, 0)],
)
def test_poly_intersection_shifted_large(shift):
    p = [(-0.5, -0.5), (99.5, -0.5), (99.5, 99.5), (-0.5, 99.5)]
    sx, sy = shift
    pq_ref = sorted(
        [
            (max(-0.5, -0.5 + sx), max(-0.5, -0.5 + sy)),
            (min(99.5, 99.5 + sx), max(-0.5, -0.5 + sy)),
            (min(99.5, 99.5 + sx), min(99.5, 99.5 + sy)),
            (max(-0.5, -0.5 + sx), min(99.5, 99.5 + sy)),
        ],
    )

    for k in range(4):
        q = [(x + sx, y + sy) for x, y in p]
        q = _roll_vertices(q, k)
        pq = clip_polygon(p, q)
        assert len(pq) == 4
        assert np.allclose(sorted(pq), pq_ref)


def test_poly_intersection_rotated45():
    p = [(0, 0), (1, 0), (1, 1), (0, 1)]
    q = [(0, 0), (SQ2, -SQ2), (2.0 * SQ2, 0), (SQ2, SQ2)]
    pq_ref = [(0, 0), (SQ2, SQ2), (1, 0), (1, SQ2 / (1.0 + SQ2))]

    for k in range(4):
        q = _roll_vertices(q, k)
        pq = clip_polygon(p, q)
        assert np.allclose(sorted(pq), pq_ref)


@pytest.mark.parametrize(
    'axis', [0, 1],
)
def test_poly_intersection_flipped_axis(axis):
    p = [(0, 0), (1, 0), (1, 1), (0, 1)]
    # (flipped wrt X-axis or Y-axis). Also change direction:
    if axis == 0:
        q = [(i, -j) for i, j in p][::-1]
    else:
        q = [(-i, j) for i, j in p][::-1]

    for k in range(4):
        q = _roll_vertices(q, k)
        pq = clip_polygon(p, q)
        assert len(pq) == 0


def test_poly_intersection_reflect_origin():
    p = [(0, 0), (1, 0), (1, 1), (0, 1)]
    # reflect wrt origin:
    q = [(-i, -j) for i, j in p]

    for k in range(4):
        q = _roll_vertices(q, k)
        pq = clip_polygon(p, q)
        assert not pq


@pytest.mark.parametrize(
    'q,small',
    [
        ([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], True),
        ([(0.0, 0.0), (1.0, 0.0), (1.0, 0.4), (0.0, 0.4)], True),
        ([(-0.1, -0.1), (1.1, -0.1), (1.1, 1.1), (-0.1, 1.1)], False),
    ],
)
def test_poly_includes_the_other(q, small):
    wnd = [(0, 0), (1, 0), (1, 1), (0, 1)]

    for k in range(4):
        q = _roll_vertices(q, k)
        qp = clip_polygon(q, wnd)

        assert _is_poly_eq(qp, q if small else wnd)


@pytest.mark.parametrize(
    'q',
    [
        [(0, 0), (1, 0), (0.5, 0.6)],
        [(0.1, 0), (0.9, 0), (0.5, 0.6)],
    ],
)
def test_poly_triangle_common_side(q):
    p = [(0, 0), (1, 0), (1, 1), (0, 1)]
    sq = sorted(q)

    for k in range(3):
        q = _roll_vertices(q, k)
        pq = clip_polygon(p, q)
        assert np.allclose(sq, sorted(pq))


def test_poly_triangle_common_side_lg():
    p = [(0, 0), (1, 0), (1, 1), (0, 1)]
    q = [(-0.1, 0), (1.1, 0), (0.5, 0.6)]
    ref_pq = [(0, 0), (0, 0.1), (0.5, 0.6), (1, 0), (1, 0.1)]

    for k in range(3):
        q = _roll_vertices(q, k)
        pq = clip_polygon(p, q)
        assert np.allclose(ref_pq, sorted(pq))


def test_poly_intersection_with_self_extra_vertices():
    p = [(0, 0), (1, 0), (1, 1), (0, 1)]
    p_ref = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # Q is same as P with extra vertices places along P's edges
    q = [(0, 0), (0.5, 0), (1, 0), (1, 0.4), (1, 1), (0.7, 1), (0, 1), (0, 0.2)]

    for k in range(4):
        q = _roll_vertices(q, k)

        pq = clip_polygon(p, q)
        assert sorted(pq) == p_ref

        pq = clip_polygon(q, p)
        assert sorted(pq) == p_ref


def test_intersection_case01():
    # a real case of failure of the code from PR #104
    p = [
        (4517.377385, 8863.424319),
        (5986.279535, 12966.888023),
        (1917.908619, 14391.538506),
        (453.893145, 10397.019260),
    ]

    wnd = [(-0.5, -0.5), (5224.5, -0.5), (5224.5, 15999.5), (-0.5, 15999.5)]

    cp_ref = [
        (4517.377385, 8863.424319),
        (5224.5, 10838.812526396974),
        (5224.5, 13233.64580022457),
        (1917.908619, 14391.538506),
        (453.893145, 10397.01926),
    ]

    cp = clip_polygon(p, wnd)

    assert _is_poly_eq(cp, cp_ref)
