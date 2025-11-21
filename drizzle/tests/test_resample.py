import math
import os
from itertools import product

import numpy as np
import pytest

from astropy import wcs
from astropy.convolution import Gaussian2DKernel
from drizzle import cdrizzle, resample, utils

from .helpers import wcs_from_file

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TEST_DIR, "data")


def bound_image(image):
    """
    Compute region where image is non-zero
    """
    coords = np.nonzero(image)
    ymin = coords[0].min()
    ymax = coords[0].max()
    xmin = coords[1].min()
    xmax = coords[1].max()
    return (ymin, ymax, xmin, xmax)


def centroid(image, size, center):
    """
    Compute the centroid of a rectangular area
    """
    ylo = int(center[0] - size / 2)
    yhi = min(ylo + size, image.shape[0])
    xlo = int(center[1] - size / 2)
    xhi = min(xlo + size, image.shape[1])

    yx1 = np.mgrid[ylo:yhi, xlo:xhi, 1:2]
    center = (yx1[..., 0] * image[ylo:yhi, xlo:xhi]).sum(
        axis=(1, 2),
        dtype=np.float64,
    )

    if center[2] == 0.0:
        return None

    center[0] /= center[2]
    center[1] /= center[2]
    return center


def centroid_close(list_of_centroids, size, point):
    """
    Find if any centroid is close to a point
    """
    for i in range(len(list_of_centroids) - 1, -1, -1):
        if abs(list_of_centroids[i][0] - point[0]) < int(size / 2) and abs(
            list_of_centroids[i][1] - point[1]
        ) < int(size / 2):
            return 1

    return 0


def centroid_compare(centroid):
    return centroid[1]


def centroid_distances(image1, image2, amp, size):
    """
    Compute a list of centroids and the distances between them in two images
    """
    distances = []
    list_of_centroids = centroid_list(image2, amp, size)
    for cc in list_of_centroids:
        center1 = centroid(image1, size, cc)
        center2 = centroid(image2, size, cc)
        if center1 is None:
            continue

        disty = center2[0] - center1[0]
        distx = center2[1] - center1[1]
        dist = math.sqrt(disty * disty + distx * distx)
        dflux = abs(center2[2] - center1[2])
        distances.append([dist, dflux, center1, center2])

    distances.sort(key=centroid_compare)
    return distances


def centroid_list(image, amp, size):
    """
    Find the next centroid
    """
    list_of_centroids = []
    points = np.transpose(np.nonzero(image > amp))
    for point in points:
        if not centroid_close(list_of_centroids, size, point):
            center = centroid(image, size, point)
            list_of_centroids.append(center)

    return list_of_centroids


def centroid_statistics(title, fname, image1, image2, amp, size):
    """
    write centroid statistics to compare differences btw two images
    """
    stats = ("minimum", "median", "maximum")
    images = (None, None, image1, image2)
    im_type = ("", "", "test", "reference")

    diff = []
    distances = centroid_distances(image1, image2, amp, size)
    indexes = (0, int(len(distances) / 2), len(distances) - 1)
    fd = open(fname, "w")
    fd.write(f"*** {title:s} ***\n")

    if len(distances) == 0:
        diff = [0.0, 0.0, 0.0]
        fd.write("No matches!!\n")

    elif len(distances) == 1:
        diff = [distances[0][0], distances[0][0], distances[0][0]]

        fd.write("1 match\n")
        fd.write(f"distance = {distances[0][0]:f} flux difference = {distances[0][1]:f}\n")

        for j in range(2, 4):
            ylo = int(distances[0][j][0]) - 1
            yhi = int(distances[0][j][0]) + 2
            xlo = int(distances[0][j][1]) - 1
            xhi = int(distances[0][j][1]) + 2
            subimage = images[j][ylo:yhi, xlo:xhi]
            fd.write(
                f"\n{im_type[j]} image centroid = "
                f"({distances[0][j][0]:f}, {distances[0][j][1]:f}) "
                f"image flux = {distances[0][j][2]:f}\n"
            )
            fd.write(str(subimage) + "\n")

    else:
        fd.write(f"{len(distances)} matches\n")

        for k in range(3):
            i = indexes[k]
            diff.append(distances[i][0])
            fd.write(
                f"\n{stats[k]} distance = {distances[i][0]:f} "
                f"flux difference = {distances[i][1]:f}\n"
            )

            for j in range(2, 4):
                ylo = int(distances[i][j][0]) - 1
                yhi = int(distances[i][j][0]) + 2
                xlo = int(distances[i][j][1]) - 1
                xhi = int(distances[i][j][1]) + 2
                subimage = images[j][ylo:yhi, xlo:xhi]
                fd.write(
                    f"\n{stats[k]} {im_type[j]} image centroid = "
                    f"({distances[i][j][0]:f}, {distances[i][j][1]:f}) "
                    f"image flux = {distances[i][j][2]:f}\n"
                )
                fd.write(str(subimage) + "\n")

    fd.close()
    return tuple(diff)


def make_point_image(shape, point, value):
    """
    Create an image with a single point set
    """
    output_image = np.zeros(shape, dtype=np.float32)
    output_image[point] = value
    return output_image


def make_grid_image(shape, spacing, value):
    """
    Create an image with points on a grid set
    """
    output_image = np.zeros(shape, dtype=np.float32)

    shape = output_image.shape
    half_space = int(spacing / 2)
    for y in range(half_space, shape[0], spacing):
        for x in range(half_space, shape[1], spacing):
            output_image[y, x] = value

    return output_image


@pytest.fixture(scope="module")
def nrcb5_stars():
    full_file_name = os.path.join(DATA_DIR, "nrcb5_sip_wcs.hdr")
    path = os.path.join(DATA_DIR, full_file_name)

    wcs, data = wcs_from_file(path, return_data=True)
    dq = np.zeros(data.shape, dtype=np.int32)
    wht = np.zeros(data.shape, dtype=np.float32)
    var = np.zeros(data.shape, dtype=np.float32)

    np.random.seed(0)

    patch_size = 21
    patch_area = patch_size**2
    p2 = patch_size // 2
    # add border so that resampled partial pixels can be isolated
    # in the segmentation:
    border = 4
    pwb = patch_size + border

    fwhm2sigma = 2.0 * math.sqrt(2.0 * math.log(2.0))

    ny, nx = data.shape

    stars = []

    for yc in range(border + p2, ny - pwb, pwb):
        for xc in range(border + p2, nx - pwb, pwb):
            sl = np.s_[yc - p2 : yc + p2 + 1, xc - p2 : xc + p2 + 1]
            flux = 1.0 + 99.0 * np.random.random()
            if np.random.random() > 0.7:
                # uniform image
                psf = np.full((patch_size, patch_size), flux / patch_area)
            else:
                # "star":
                fwhm = 1.5 + 1.5 * np.random.random()
                sigma = fwhm / fwhm2sigma

                psf = flux * Gaussian2DKernel(sigma, x_size=patch_size, y_size=patch_size).array
            weight = 0.6 + 0.4 * np.random.random((patch_size, patch_size))
            wflux = (psf * weight).sum()

            mean_noise = (0.05 + 0.35 * np.random.random()) * flux / patch_area
            rdnoise = mean_noise * np.random.random((patch_size, patch_size))

            data[sl] = psf
            wht[sl] = weight
            dq[sl] = 0
            var_patch = psf + rdnoise
            var[sl] = var_patch
            stars.append((xc, yc, wflux, (var_patch * weight**2).sum(), sl))

    return data, wht, dq, var, stars, wcs


def test_drizzle_defaults():
    n = 200
    in_shape = (n, n)

    # input coordinate grid:
    y, x = np.indices(in_shape, dtype=np.float64)

    # simulate data:
    in_sci = np.ones(in_shape, dtype=np.float32)
    in_wht = np.ones(in_shape, dtype=np.float32)

    # create a Drizzle object using all default parameters (except for 'kernel')
    driz = resample.Drizzle(
        kernel="square",
    )

    assert driz.out_img is None
    assert driz.out_wht is None
    assert driz.out_ctx is None
    assert driz.total_exptime == 0.0

    driz.add_image(
        in_sci,
        exptime=1.0,
        pixmap=np.dstack([x, y]),
        weight_map=in_wht,
    )

    pixmap = np.dstack([x + 1, y + 2])
    driz.add_image(
        3 * in_sci,
        exptime=1.0,
        pixmap=pixmap,
        weight_map=in_wht,
    )

    assert driz.out_img[0, 0] == 1
    assert driz.out_img[1, 0] == 1
    assert driz.out_img[2, 0] == 1
    assert driz.out_img[1, 1] == 1
    assert driz.out_img[1, 2] == 1
    assert (driz.out_img[2, 1] - 2.0) < 1.0e-14


@pytest.mark.parametrize(
    "kernel,test_image_type,max_diff_atol",
    [
        ("square", "point", 1.0e-5),
        ("square", "grid", 1.0e-5),
        ("point", "grid", 1.0e-5),
        ("turbo", "grid", 1.0e-5),
        ("lanczos3", "grid", 1.0e-5),
        ("gaussian", "grid", 2.0e-5),
    ],
)
def test_resample_kernel(tmpdir, kernel, test_image_type, max_diff_atol):
    """
    Test do_driz square kernel with point
    """
    output_difference = str(tmpdir.join(f"difference_{kernel}_{test_image_type}.txt"))

    inwcs = wcs_from_file("j8bt06nyq_flt.fits", ext=1)
    if test_image_type == "point":
        insci = make_point_image(inwcs.array_shape, (500, 200), 100.0)
    else:
        insci = make_grid_image(inwcs.array_shape, 64, 100.0)
    inwht = np.ones_like(insci)
    output_wcs, template_data = wcs_from_file(
        f"reference_{kernel}_{test_image_type}.fits", ext=1, return_data=True
    )

    pixmap = utils.calc_pixmap(
        inwcs,
        output_wcs,
    )

    if kernel == "point":
        pscale_ratio = 1.0
    else:
        pscale_ratio = utils.estimate_pixel_scale_ratio(
            inwcs,
            output_wcs,
            refpix_from=inwcs.wcs.crpix,
            refpix_to=output_wcs.wcs.crpix,
        )

        # ignore previous pscale and compute it the old way (only to make
        # tests work with old truth files and thus to show that new API gives
        # same results when equal definitions of the pixel scale is used):
        pscale_ratio = np.sqrt(
            np.sum(output_wcs.wcs.pc**2, axis=0)[0] / np.sum(inwcs.wcs.cd**2, axis=0)[0]
        )

    driz = resample.Drizzle(
        kernel=kernel,
        out_shape=output_wcs.array_shape,
        fillval=0.0,
    )

    if kernel in ["square", "turbo", "point"]:
        driz.add_image(
            insci,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=inwht,
            iscale=pscale_ratio**2,
            pixel_scale_ratio=pscale_ratio,
        )
    else:
        with pytest.warns(Warning, match=f"Kernel '{kernel}' is not a flux-conserving kernel"):
            driz.add_image(
                insci,
                exptime=1.0,
                pixmap=pixmap,
                weight_map=inwht,
                iscale=pscale_ratio**2,
                pixel_scale_ratio=pscale_ratio,
            )

    _, med_diff, max_diff = centroid_statistics(
        f"{kernel} with {test_image_type}",
        output_difference,
        driz.out_img,
        template_data,
        30.0,
        8,
    )

    assert med_diff < 1.0e-6
    assert max_diff < max_diff_atol


@pytest.mark.parametrize(
    "kernel,max_diff_atol",
    [
        ("square", 1.0e-5),
        ("turbo", 1.0e-5),
    ],
)
def test_resample_kernel_image(tmpdir, kernel, max_diff_atol):
    """
    Test do_driz square kernel with point
    """
    inwcs, insci = wcs_from_file("j8bt06nyq_flt.fits", ext=1, return_data=True)
    inwht = np.ones_like(insci)

    outwcs, ref_sci, ref_ctx, ref_wht = wcs_from_file(
        f"reference_{kernel}_image.fits", ext=1, return_data=["SCI", "CTX", "WHT"]
    )
    ref_ctx = np.array(ref_ctx, dtype=np.int32)

    pixmap = utils.calc_pixmap(
        inwcs,
        outwcs,
    )

    pscale_ratio = np.sqrt(np.sum(outwcs.wcs.cd**2, axis=0)[0] / np.sum(inwcs.wcs.cd**2, axis=0)[0])

    driz = resample.Drizzle(
        kernel=kernel,
        out_shape=ref_sci.shape,
        fillval=0.0,
    )

    driz.add_image(
        insci,
        exptime=1.0,
        pixmap=pixmap,
        weight_map=inwht,
        iscale=pscale_ratio**2,
        pixel_scale_ratio=pscale_ratio,
    )
    outctx = driz.out_ctx[0]

    # in order to avoid small differences in the staircase in the outline
    # of the input image in the output grid, select a subset:
    sl = np.s_[125:-125, 5:-5]

    assert np.allclose(driz.out_img[sl], ref_sci[sl], atol=0, rtol=1.0e-6)
    assert np.allclose(driz.out_wht[sl], ref_wht[sl], atol=0, rtol=1.0e-6)
    assert np.all(outctx[sl] == ref_ctx[sl])


@pytest.mark.parametrize(
    "kernel,fc",
    [
        ("square", True),
        ("point", True),
        ("turbo", True),
        ("lanczos2", False),
        ("lanczos3", False),
        ("gaussian", False),
    ],
)
def test_zero_input_weight(kernel, fc):
    """
    Test do_driz square kernel with grid
    """
    # initialize input:
    insci = np.ones((200, 400), dtype=np.float32)
    inwht = np.ones((200, 400), dtype=np.float32)
    inwht[:, 150:155] = 0

    # initialize output:
    outsci = np.zeros((210, 410), dtype=np.float32)
    outwht = np.zeros((210, 410), dtype=np.float32)
    outctx = np.zeros((210, 410), dtype=np.int32)

    # define coordinate mapping:
    pixmap = np.moveaxis(np.mgrid[1:201, 1:401][::-1], 0, -1)

    # resample:
    if fc:
        cdrizzle.tdriz(
            insci,
            inwht,
            pixmap,
            outsci,
            outwht,
            outctx,
            uniqid=1,
            xmin=0,
            xmax=400,
            ymin=0,
            ymax=200,
            pixfrac=1,
            kernel=kernel,
            in_units="cps",
            expscale=1,
            wtscale=1,
            fillstr="INDEF",
        )
    else:
        with pytest.warns(Warning, match=f"Kernel '{kernel}' is not a flux-conserving kernel"):
            cdrizzle.tdriz(
                insci,
                inwht,
                pixmap,
                outsci,
                outwht,
                outctx,
                uniqid=1,
                xmin=0,
                xmax=400,
                ymin=0,
                ymax=200,
                pixfrac=1,
                kernel=kernel,
                in_units="cps",
                expscale=1,
                wtscale=1,
                fillstr="INDEF",
            )
        # pytest.xfail("Not a flux-conserving kernel")

    # check that no pixel with 0 weight has any counts:
    assert np.sum(np.abs(outsci[(outwht == 0)])) == 0.0


@pytest.mark.parametrize(
    "interpolator,test_image_type",
    [
        ("poly5", "point"),
        ("default", "grid"),
        ("lan3", "grid"),
        ("lan5", "grid"),
    ],
)
def test_blot_interpolation(tmpdir, interpolator, test_image_type):
    """
    Test do_driz square kernel with point
    """
    output_difference = str(tmpdir.join(f"difference_blot_{interpolator}_{test_image_type}.txt"))

    outwcs = wcs_from_file("j8bt06nyq_flt.fits", ext=1)
    if test_image_type == "point":
        outsci = make_point_image(outwcs.array_shape, (500, 200), 40.0)
        ref_fname = "reference_blot_point.fits"
    else:
        outsci = make_grid_image(outwcs.array_shape, 64, 100.0)
        ref_fname = f"reference_blot_{interpolator}.fits"
    inwcs, template_data = wcs_from_file(ref_fname, ext=1, return_data=True)

    pixmap = utils.calc_pixmap(inwcs, outwcs)

    # compute pscale the old way (only to make
    # tests work with old truth files and thus to show that new API gives
    # same results when equal definitions of the pixel scale is used):
    pscale_ratio = np.sqrt(np.sum(inwcs.wcs.pc**2, axis=0)[0] / np.sum(outwcs.wcs.cd**2, axis=0)[0])

    if interpolator == "default":
        kwargs = {}
    else:
        kwargs = {"interp": interpolator}

    blotted_image = resample.blot_image(
        outsci, pixmap=pixmap, iscale=1.0 / (pscale_ratio**2), **kwargs
    )

    _, med_diff, max_diff = centroid_statistics(
        "blot with '{interpolator}' and '{test_image_type}'",
        output_difference,
        blotted_image,
        template_data,
        20.0,
        16,
    )
    assert med_diff < 1.0e-6
    assert max_diff < 1.0e-5


def test_context_planes():
    """Reproduce error seen in issue #50"""
    shape = (10, 10)
    output_wcs = wcs.WCS()
    output_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    output_wcs.wcs.pc = [[1, 0], [0, 1]]
    output_wcs.pixel_shape = shape
    driz = resample.Drizzle(out_shape=tuple(shape))

    image = np.ones(shape)
    inwcs = wcs.WCS()
    inwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    inwcs.wcs.cd = [[1, 0], [0, 1]]
    inwcs.pixel_shape = shape

    pixmap = utils.calc_pixmap(inwcs, output_wcs)

    # context image must be 2D or 3D:
    with pytest.raises(ValueError) as err_info:
        resample.Drizzle(
            kernel="point",
            exptime=0.0,
            out_shape=shape,
            out_ctx=[0, 0, 0],
        )
    assert str(err_info.value).startswith("'out_ctx' must be either a 2D or 3D array.")

    driz = resample.Drizzle(
        kernel="square",
        out_shape=output_wcs.array_shape,
        fillval=0.0,
    )

    for i in range(32):
        assert driz.ctx_id == i
        driz.add_image(image, exptime=1.0, pixmap=pixmap)
    assert driz.out_ctx.shape == (1, 10, 10)

    driz.add_image(image, exptime=1.0, pixmap=pixmap)
    assert driz.out_ctx.shape == (2, 10, 10)


def test_no_context_image():
    """Reproduce error seen in issue #50"""
    shape = (10, 10)
    output_wcs = wcs.WCS()
    output_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    output_wcs.wcs.pc = [[1, 0], [0, 1]]
    output_wcs.pixel_shape = shape
    driz = resample.Drizzle(out_shape=tuple(shape), begin_ctx_id=-1, disable_ctx=True)
    assert driz.out_ctx is None
    assert driz.ctx_id is None

    image = np.ones(shape)
    inwcs = wcs.WCS()
    inwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    inwcs.wcs.cd = [[1, 0], [0, 1]]
    inwcs.pixel_shape = shape

    pixmap = utils.calc_pixmap(inwcs, output_wcs)

    for i in range(33):
        driz.add_image(image, exptime=1.0, pixmap=pixmap)
        assert driz.out_ctx is None
        assert driz.ctx_id is None


def test_init_ctx_id():
    # starting context ID must be positive
    with pytest.raises(ValueError) as err_info:
        resample.Drizzle(
            kernel="square",
            exptime=0.0,
            begin_ctx_id=-1,
            out_shape=(10, 10),
        )
    assert str(err_info.value).startswith("Invalid context image ID")

    with pytest.raises(ValueError) as err_info:
        resample.Drizzle(
            kernel="square",
            exptime=0.0,
            out_shape=(10, 10),
            begin_ctx_id=1,
            max_ctx_id=0,
        )
    assert str(err_info.value).startswith("'max_ctx_id' cannot be smaller than 'begin_ctx_id'.")


def test_context_agrees_with_weight():
    n = 200
    out_shape = (n, n)

    # allocate output arrays:
    out_img = np.zeros(out_shape, dtype=np.float32)
    out_ctx = np.zeros(out_shape, dtype=np.int32)
    out_wht = np.zeros(out_shape, dtype=np.float32)

    # previous data in weight and context must agree:
    with pytest.raises(ValueError) as err_info:
        out_ctx[0, 0] = 1
        out_ctx[0, 1] = 1
        out_wht[0, 0] = 0.1
        resample.Drizzle(
            kernel="square",
            out_shape=out_shape,
            out_img=out_img,
            out_ctx=out_ctx,
            out_wht=out_wht,
            exptime=1.0,
        )
    assert str(err_info.value).startswith(
        "Inconsistent values of supplied 'out_wht' and 'out_ctx' "
    )


@pytest.mark.parametrize(
    "kernel,fc,pixel_scale_ratio",
    [
        ("square", True, 1.0),
        ("point", True, 1.0),
        ("turbo", True, 1.0),
        ("turbo", True, None),
        ("lanczos2", False, 1.0),
        ("lanczos2", False, None),
        ("lanczos3", False, 1.0),
        ("lanczos3", False, None),
        ("gaussian", False, 1.0),
        ("gaussian", False, None),
    ],
)
def test_flux_conservation_nondistorted(kernel, fc, pixel_scale_ratio):
    n = 200
    in_shape = (n, n)

    # input coordinate grid:
    y, x = np.indices(in_shape, dtype=np.float64)

    # simulate a gaussian "star":
    fwhm = 2.9
    x0 = 50.0
    y0 = 68.0
    sig = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0 * fwhm)))
    sig2 = sig * sig
    star = np.exp(
        -0.5 / sig2 * ((x.astype(np.float32) - x0) ** 2 + (y.astype(np.float32) - y0) ** 2)
    )
    in_sci = (star / np.sum(star)).astype(np.float32)  # normalize to 1
    in_wht = np.ones(in_shape, dtype=np.float32)

    # linear shift:
    xp = x + 0.5
    yp = y + 0.2

    pixmap = np.dstack([xp, yp])

    out_shape = (int(yp.max()) + 1, int(xp.max()) + 1)
    # make sure distorion is not moving flux out of the image towards negative
    # coordinates (just because of the simple way of how we account for output
    # image size)
    assert np.min(xp) > -0.5 and np.min(yp) > -0.5

    out_img = np.zeros(out_shape, dtype=np.float32)
    out_ctx = np.zeros(out_shape, dtype=np.int32)
    out_wht = np.zeros(out_shape, dtype=np.float32)

    if fc:
        cdrizzle.tdriz(
            in_sci,
            in_wht,
            pixmap,
            out_img,
            out_wht,
            out_ctx,
            pixfrac=1.0,
            pscale_ratio=pixel_scale_ratio,
            kernel=kernel,
            in_units="cps",
            expscale=1.0,
            wtscale=1.0,
        )
    else:
        with pytest.warns(Warning, match=f"Kernel '{kernel}' is not a flux-conserving kernel"):
            cdrizzle.tdriz(
                in_sci,
                in_wht,
                pixmap,
                out_img,
                out_wht,
                out_ctx,
                pixfrac=1.0,
                pscale_ratio=pixel_scale_ratio,
                kernel=kernel,
                in_units="cps",
                expscale=1.0,
                wtscale=1.0,
            )

        pytest.xfail("Not a flux-conserving kernel")

    assert np.allclose(
        np.sum(out_img * out_wht),
        np.sum(in_sci),
        atol=0.0,
        rtol=0.0001,
    )


@pytest.mark.parametrize(
    "kernel,fc",
    [
        ("square", True),
        ("point", True),
        ("turbo", True),
        ("lanczos2", False),
        ("lanczos3", False),
        ("gaussian", False),
    ],
)
def test_flux_conservation_distorted(kernel, fc):
    n = 200
    in_shape = (n, n)

    # input coordinate grid:
    y, x = np.indices(in_shape, dtype=np.float64)

    # simulate a gaussian "star":
    fwhm = 2.9
    x0 = 50.0
    y0 = 68.0
    sig = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0 * fwhm)))
    sig2 = sig * sig
    star = np.exp(
        -0.5 / sig2 * ((x.astype(np.float32) - x0) ** 2 + (y.astype(np.float32) - y0) ** 2)
    )
    in_sci = (star / np.sum(star)).astype(np.float32)  # normalize to 1
    in_wht = np.ones(in_shape, dtype=np.float32)

    # linear shift:
    xp = x + 0.5
    yp = y + 0.2
    # add distortion:
    xp += 1e-4 * x**2 + 1e-5 * x * y
    yp += 1e-3 * y**2 - 2e-5 * x * y

    pixmap = np.dstack([xp, yp])

    out_shape = (int(yp.max()) + 1, int(xp.max()) + 1)
    # make sure distorion is not moving (pixels with) flux out of the image
    # towards negative coordinates (just because of the simple way of how we
    # account for output image size):
    assert np.min(xp) > -0.5 and np.min(yp) > -0.5

    out_img = np.zeros(out_shape, dtype=np.float32)
    out_ctx = np.zeros(out_shape, dtype=np.int32)
    out_wht = np.zeros(out_shape, dtype=np.float32)

    if fc:
        cdrizzle.tdriz(
            in_sci,
            in_wht,
            pixmap,
            out_img,
            out_wht,
            out_ctx,
            pixfrac=1.0,
            pscale_ratio=1.0,
            kernel=kernel,
            in_units="cps",
            expscale=1.0,
            wtscale=1.0,
        )
    else:
        with pytest.warns(Warning, match=f"Kernel '{kernel}' is not a flux-conserving kernel"):
            cdrizzle.tdriz(
                in_sci,
                in_wht,
                pixmap,
                out_img,
                out_wht,
                out_ctx,
                pixfrac=1.0,
                pscale_ratio=1.0,
                kernel=kernel,
                in_units="cps",
                expscale=1.0,
                wtscale=1.0,
            )
        pytest.xfail("Not a flux-conserving kernel")

    assert np.allclose(
        np.sum(out_img * out_wht),
        np.sum(in_sci),
        atol=0.0,
        rtol=0.0001,
    )


@pytest.mark.parametrize("kernel", ["square", "turbo", "point"])
@pytest.mark.parametrize("pscale_ratio", [0.55, 1.0, 1.2])
def test_flux_conservation_distorted_distributed_sources(nrcb5_stars, kernel, pscale_ratio):
    """test aperture photometry"""
    insci, inwht, _, invar, stars, wcs = nrcb5_stars

    suffix = f"{pscale_ratio}".replace(".", "p")
    output_wcs = wcs_from_file(f"nrcb5_output_wcs_psr_{suffix}.hdr")

    pixmap = utils.calc_pixmap(
        wcs,
        output_wcs,
        wcs.array_shape,
    )

    # resample variance using squared coefficients AND a "point" kernel
    # ("point" kernel is needed for the *aperture* photometry/variance)
    driz_var = resample.Drizzle(
        kernel="point",
        out_shape=output_wcs.array_shape,
        fillval=0.0,
    )
    driz_var.add_image(
        data=insci,
        data2=invar,
        exptime=1.0,
        pixmap=pixmap,
        weight_map=inwht,
        iscale=1.0,
        pixel_scale_ratio=1.0,
    )

    # for efficiency, instead of doing this patch-by-patch,
    # multiply resampled data by resampled image weight
    if kernel == "point":
        out_data = driz_var.out_img * driz_var.out_wht
    else:
        # resample "SCI" array using user-specified kernel:
        driz = resample.Drizzle(
            kernel=kernel,
            out_shape=output_wcs.array_shape,
            fillval=0.0,
        )
        driz.add_image(
            data=insci,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=inwht,
            iscale=1.0,
            pixel_scale_ratio=1.0,
        )
        out_data = driz_var.out_img * driz_var.out_wht
    out_var = driz_var.out_img2[0] * (driz_var.out_wht**2)

    dim3 = (slice(None, None, None),)

    for _, _, wfin, wvfin, sl in stars:
        xyout = pixmap[sl + dim3]
        xmin = math.floor(xyout[:, :, 0].min() - 0.5)
        xmax = math.ceil(xyout[:, :, 0].max() + 1.5)
        ymin = math.floor(xyout[:, :, 1].min() - 0.5)
        ymax = math.ceil(xyout[:, :, 1].max() + 1.5)

        wfout = np.nansum(out_data[ymin:ymax, xmin:xmax])
        wvfout = np.nansum(out_var[ymin:ymax, xmin:xmax])

        # test resampled "weighted" flux in an aperture matches flux from input
        assert np.allclose(wfin, wfout, rtol=1.0e-6, atol=0.0)

        # test resampled "weighted" variance in an aperture matches
        # "weighted" variance from input
        assert np.allclose(wvfin, wvfout, rtol=1.0e-6, atol=0.0)


def test_drizzle_exptime():
    n = 200
    in_shape = (n, n)

    # input coordinate grid:
    y, x = np.indices(in_shape, dtype=np.float64)

    # simulate data:
    in_sci = np.ones(in_shape, dtype=np.float32)
    in_wht = np.ones(in_shape, dtype=np.float32)

    pixmap = np.dstack([x, y])

    # allocate output arrays:
    out_shape = (int(y.max()) + 1, int(x.max()) + 1)
    out_img = np.zeros(out_shape, dtype=np.float32)
    out_ctx = np.zeros(out_shape, dtype=np.int32)
    out_wht = np.zeros(out_shape, dtype=np.float32)

    # starting exposure time must be non-negative:
    with pytest.raises(ValueError) as err_info:
        driz = resample.Drizzle(
            kernel="square",
            out_shape=out_shape,
            fillval="indef",
            exptime=-1.0,
        )
    assert str(err_info.value) == "Exposure time must be non-negative."

    driz = resample.Drizzle(
        kernel="turbo",
        out_shape=out_shape,
        fillval="",
        out_img=out_img,
        out_ctx=out_ctx,
        out_wht=out_wht,
        exptime=1.0,
    )
    assert driz.kernel == "turbo"

    driz.add_image(in_sci, weight_map=in_wht, exptime=1.03456, pixmap=pixmap)
    assert np.allclose(driz.total_exptime, 2.03456, rtol=0, atol=1.0e-14)

    driz.add_image(in_sci, weight_map=in_wht, exptime=3.1415926, pixmap=pixmap)
    assert np.allclose(driz.total_exptime, 5.1761526, rtol=0, atol=1.0e-14)

    with pytest.raises(ValueError) as err_info:
        driz.add_image(in_sci, weight_map=in_wht, exptime=-1, pixmap=pixmap)
    assert str(err_info.value) == "'exptime' *must* be a strictly positive number."

    # exptime cannot be 0 when output data has data:
    with pytest.raises(ValueError) as err_info:
        out_ctx[0, 0] = 1
        driz = resample.Drizzle(
            kernel="square",
            out_shape=out_shape,
            fillval="indef",
            out_img=out_img,
            out_ctx=out_ctx,
            out_wht=out_wht,
            exptime=0.0,
        )
    assert str(err_info.value).startswith(
        "Inconsistent exposure time and context and/or weight images:"
    )

    # exptime must be 0 when output arrays are not provided:
    with pytest.raises(ValueError) as err_info:
        driz = resample.Drizzle(
            kernel="square",
            out_shape=out_shape,
            exptime=1.0,
        )
    assert str(err_info.value).startswith("Exposure time must be 0.0 for the first resampling")


def test_drizzle_unsupported_kernel():
    with pytest.raises(ValueError) as err_info:
        resample.Drizzle(
            kernel="magic_image_improver",
            out_shape=(10, 10),
            exptime=0.0,
        )
    assert str(err_info.value) == "Kernel 'magic_image_improver' is not supported."


def test_pixmap_shape_matches_image():
    n = 200
    in_shape = (n, n)

    # input coordinate grid:
    y, x = np.indices((n + 1, n), dtype=np.float64)

    # simulate data:
    in_sci = np.ones(in_shape, dtype=np.float32)
    in_wht = np.ones(in_shape, dtype=np.float32)

    pixmap = np.dstack([x, y])

    driz = resample.Drizzle(
        kernel="square",
        fillval=0.0,
        exptime=0.0,
    )

    # last two sizes of the pixelmap must match those of input images:
    with pytest.raises(ValueError) as err_info:
        driz.add_image(
            in_sci,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht,
            iscale=1.0,
            pixel_scale_ratio=1.0,
        )
    assert str(err_info.value) == "'pixmap' shape is not consistent with 'data' shape."


def test_drizzle_fillval():
    n = 200
    in_shape = (n, n)

    # input coordinate grid:
    y, x = np.indices(in_shape, dtype=np.float64)

    # simulate a gaussian "star":
    fwhm = 2.9
    x0 = 50.0
    y0 = 68.0
    sig = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0 * fwhm)))
    sig2 = sig * sig
    star = np.exp(
        -0.5 / sig2 * ((x.astype(np.float32) - x0) ** 2 + (y.astype(np.float32) - y0) ** 2)
    )
    in_sci = (star / np.sum(star)).astype(np.float32)  # normalize to 1
    in_wht = np.zeros(in_shape, dtype=np.float32)
    mask = np.where((x.astype(np.float32) - x0) ** 2 + (y.astype(np.float32) - y0) ** 2 <= 10)
    in_wht[mask] = 1.0

    # linear shift:
    xp = x + 50
    yp = y + 50

    pixmap = np.dstack([xp, yp])

    out_shape = (int(yp.max()) + 1, int(xp.max()) + 1)
    # make sure distorion is not moving flux out of the image towards negative
    # coordinates (just because of the simple way of how we account for output
    # image size)
    assert np.min(xp) > -0.5 and np.min(yp) > -0.5

    out_img = np.zeros(out_shape, dtype=np.float32) - 1.11
    out_ctx = np.zeros((1,) + out_shape, dtype=np.int32)
    out_wht = np.zeros(out_shape, dtype=np.float32)

    driz = resample.Drizzle(
        kernel="square",
        out_shape=out_shape,
        fillval="indef",
        exptime=0.0,
    )

    driz.add_image(in_sci, weight_map=in_wht, exptime=1.0, pixmap=pixmap)
    assert np.isnan(driz.out_img[0, 0])
    assert driz.out_img[int(y0) + 50, int(x0) + 50] > 0.0

    driz = resample.Drizzle(
        kernel="square",
        out_shape=out_shape,
        fillval="-1.11",
        out_img=out_img.copy(),
        out_ctx=out_ctx.copy(),
        out_wht=out_wht.copy(),
        exptime=0.0,
    )
    driz.add_image(in_sci, weight_map=in_wht, exptime=1.0, pixmap=pixmap)
    assert np.allclose(driz.out_img[0, 0], -1.11, rtol=0.0, atol=1.0e-7)
    assert driz.out_img[int(y0) + 50, int(x0) + 50] > 0.0
    assert set(driz.out_ctx.ravel().tolist()) == {0, 1}

    # test same with numeric fillval:
    driz = resample.Drizzle(
        kernel="square",
        out_shape=out_shape,
        fillval=-1.11,
        out_img=out_img.copy(),
        out_ctx=out_ctx.copy(),
        out_wht=out_wht.copy(),
        exptime=0.0,
    )
    driz.add_image(in_sci, weight_map=in_wht, exptime=1.0, pixmap=pixmap)
    assert np.allclose(driz.out_img[0, 0], -1.11, rtol=0.0, atol=1.0e-7)
    assert np.allclose(float(driz.fillval), -1.11, rtol=0.0, atol=np.finfo(float).eps)

    # make sure code raises exception for unsupported fillval:
    with pytest.raises(ValueError) as err_info:
        resample.Drizzle(
            kernel="square",
            out_shape=out_shape,
            fillval="fillval",
            exptime=0.0,
        )
    assert str(err_info.value) == "could not convert string to float: 'fillval'"


def test_resample_get_shape_from_pixmap():
    n = 200
    in_shape = (n, n)

    # input coordinate grid:
    y, x = np.indices(in_shape, dtype=np.float64)

    # simulate constant data:
    in_sci = np.ones(in_shape, dtype=np.float32)
    in_wht = np.ones(in_shape, dtype=np.float32)

    pixmap = np.dstack([x, y])

    driz = resample.Drizzle(
        kernel="point",
        exptime=0.0,
    )

    driz.add_image(in_sci, weight_map=in_wht, exptime=0.1, pixmap=pixmap)
    assert driz.out_img.shape == in_shape


def test_resample_counts_units():
    n = 200
    in_shape = (n, n)

    # input coordinate grid:
    y, x = np.indices(in_shape, dtype=np.float64)
    pixmap = np.dstack([x, y])

    # simulate constant data:
    in_sci = np.ones(in_shape, dtype=np.float32)
    in_wht = np.ones(in_shape, dtype=np.float32)

    driz = resample.Drizzle()
    driz.add_image(in_sci, weight_map=in_wht, exptime=1.0, pixmap=pixmap, in_units="cps")
    cps_max_val = driz.out_img.max()

    driz = resample.Drizzle()
    driz.add_image(in_sci, weight_map=in_wht, exptime=2.0, pixmap=pixmap, in_units="counts")
    counts_max_val = driz.out_img.max()

    assert abs(counts_max_val - cps_max_val / 2.0) < 1.0e-14


def test_resample_inconsistent_output():
    n = 200
    out_shape = (n, n)

    # different shapes:
    out_img = np.zeros((n, n), dtype=np.float32)
    out_ctx = np.zeros((1, n, n + 1), dtype=np.int32)
    out_wht = np.zeros((n + 1, n + 1), dtype=np.float32)

    # shape from out_img:
    driz = resample.Drizzle(
        kernel="point",
        exptime=0.0,
        out_img=out_img,
    )
    assert driz.out_img.shape == out_shape

    # inconsistent shapes:
    out_shape = (n + 1, n)
    with pytest.raises(ValueError) as err_info:
        resample.Drizzle(
            kernel="point",
            exptime=0.0,
            out_shape=out_shape,
            out_img=out_img,
            out_ctx=out_ctx,
            out_wht=out_wht,
        )
    assert str(err_info.value).startswith("Inconsistent data shapes specified")


def test_resample_disable_ctx():
    n = 20
    in_shape = (n, n)

    pixmap = np.dstack(np.indices(in_shape, dtype=np.float64)[::-1])

    # simulate constant data:
    in_sci = np.ones(in_shape, dtype=np.float32)

    driz = resample.Drizzle(
        disable_ctx=True,
    )

    driz.add_image(in_sci, exptime=1.0, pixmap=pixmap)


@pytest.mark.parametrize("fillval", ["NaN", "INDEF", "", None])
def test_nan_fillval(fillval):
    driz = resample.Drizzle(kernel="square", fillval=fillval, out_shape=(20, 20))

    assert np.all(np.isnan(driz.out_img))


def test_resample_edge_sgarea_bug():
    """
    Test from https://github.com/spacetelescope/drizzle/issues/187

    """
    pixmap = np.array(
        [
            [
                [0.31887051, 1.0],
                [1.01898591, 1.0],
                [1.71909665, 1.0],
            ],
            [
                [0.31591881, 0.0],
                [1.0160342312345672, 0.0],
                [1.716145, 0.0],
            ],
        ],
        dtype="f8",
    )

    in_shape = pixmap.shape[:2]
    img = np.full(in_shape, 42, dtype=np.float32)
    out_shape = (4, 4)

    driz = resample.Drizzle(
        kernel="square",
        fillval="nan",
        out_shape=out_shape,
        disable_ctx=True,
    )

    driz.add_image(
        img,
        exptime=11.776,
        in_units="cps",
        pixfrac=1.0,
        pixmap=pixmap,
        iscale=1.0,
        pixel_scale_ratio=1.0,
        wht_scale=1.0,
    )
    # expected pixels should be close to 42
    np.testing.assert_allclose(driz.out_img[:2, :3], img[0, 0], rtol=1e-6)

    # other values should be nan
    np.testing.assert_equal(driz.out_img[:, 3:], np.nan)
    np.testing.assert_equal(driz.out_img[2:], np.nan)


def test_resample_edge_collinear():
    """
    Test that resample does not crash when the input image is smaller than the
    output image, and the edges of the two images are nearly collinear.

    Test based on the example from
    https://github.com/spacetelescope/drizzle/issues/189#issue-3196294879

    """
    pixmap = np.array(
        [
            [
                [0.31, 1.0],
                [1.01, 1.0],
                [2.01, 1.0],
            ],
            [
                [0.31, 0.0],
                [1.01, 0.0],
                [1.71, 0.0],
            ],
        ],
        dtype="f8",
    )

    in_shape = pixmap.shape[:2]
    img = np.full(in_shape, np.pi, dtype=np.float32)
    in_flux = np.sum(img)
    out_shape = (4, 4)

    driz = resample.Drizzle(
        kernel="square",
        fillval="nan",
        out_shape=out_shape,
        disable_ctx=True,
    )

    driz.add_image(
        img,
        exptime=11.776,
        in_units="cps",
        pixfrac=1.0,
        pixmap=pixmap,
        iscale=1.0,
        pixel_scale_ratio=1.0,
        wht_scale=1.0,
    )

    out_flux = np.nansum(driz.out_img * driz.out_wht)

    # Given this pixmap, the entire input image should fit within the output
    # image. There should be at least 7 pixels with finite values in the output
    # image. We can get more than 7 pixels with finite values due to rounding
    # errors when computing polygon intersections (those "extra" pixels should)
    # have very small weights.
    assert np.sum(driz.out_wht > 1e-30) == 7
    assert np.sum(np.isfinite(driz.out_img)) >= 7
    # output image intensity must be equal to the input image intensity:
    assert np.allclose(driz.out_img[np.isfinite(driz.out_img)], img[0, 0], rtol=0, atol=1e-6)
    # flux in the output image should be equal to the flux in the input image:
    assert np.allclose(out_flux, in_flux, rtol=1e-6, atol=0.0)
    # area of the signal in the input image:
    assert np.allclose(np.sum(driz.out_wht), 6.0, rtol=0, atol=1.0e-6)


@pytest.mark.parametrize(
    "kernel,fc",
    [
        ("square", True),
        ("point", True),
        ("turbo", True),
        ("lanczos2", False),
        ("lanczos3", False),
        ("gaussian", False),
    ],
)
def test_drizzle_weights_squared(kernel, fc):
    n = 17
    in_shape = (n, n)

    # input coordinate grid:
    y, x = np.indices(in_shape, dtype=np.float64)

    in_sci1 = np.zeros(in_shape, dtype=np.float32)
    in_wht1 = np.zeros(in_shape, dtype=np.float32)
    in_sci1_sq = np.zeros(in_shape, dtype=np.float32)

    in_sci2 = np.zeros(in_shape, dtype=np.float32)
    in_wht2 = np.zeros(in_shape, dtype=np.float32)
    in_sci2_sq = np.zeros(in_shape, dtype=np.float32)

    xc = yc = n // 2

    in_sci1[yc, xc] = 1.0
    in_wht1[yc, xc] = 0.99
    in_sci1_sq[yc, xc] = 0.5

    in_sci2[yc, xc] = 7.0
    in_wht2[yc, xc] = 0.01
    in_sci2_sq[yc, xc] = 50.0

    pixmap = np.dstack([x, y])

    out_shape = (int(y.max()) + 1, int(x.max()) + 1)

    if fc:
        # create a Drizzle object using all default parameters
        # (except for 'kernel', 'out_shape')
        driz = resample.Drizzle(
            kernel=kernel,
            out_shape=out_shape,
            fillval2=-99.0,
        )

        assert driz.out_img is not None
        assert driz.out_img2 is None
        assert driz.total_exptime == 0.0

        driz.add_image(
            data=in_sci1,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht1,
            data2=[in_sci1_sq],
        )

        driz.add_image(
            data=in_sci2,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht2,
            data2=in_sci2_sq,
        )

        assert isinstance(driz.out_img2, list)
        assert len(driz.out_img2) == 1

    else:
        # create a Drizzle object using mostly default parameters
        driz = resample.Drizzle(
            kernel=kernel,
            out_img2=[None],
            fillval2=-99.0,
        )

        assert driz.out_img is None
        assert driz.total_exptime == 0.0

        with pytest.warns(Warning, match=f"Kernel '{kernel}' is not a flux-conserving kernel"):
            driz.add_image(
                data=in_sci1,
                exptime=1.0,
                pixmap=pixmap,
                weight_map=in_wht1,
                data2=[in_sci1_sq],
            )
        with pytest.warns(Warning, match=f"Kernel '{kernel}' is not a flux-conserving kernel"):
            driz.add_image(
                data=in_sci2,
                exptime=1.0,
                pixmap=pixmap,
                weight_map=in_wht2,
                data2=[in_sci2_sq],
            )

    assert np.allclose(np.max(driz.out_img2), 0.495050013, rtol=1.0e-6, atol=0.0)

    # check fill value
    assert np.allclose(np.min(driz.out_img2), -99.0, rtol=1.0e-6, atol=0.0)
    assert abs(float(driz.fillval2) + 99.0) < 1e-7


@pytest.mark.filterwarnings("ignore:Kernel '")
@pytest.mark.parametrize(
    "kernel_fc, pscale, weights",
    (
        x
        for x in product(
            [
                ("square", True),
                ("turbo", True),
                ("point", True),
                ("gaussian", False),
                # lanczos kernels do not support pscale != 1 or pixfrac != 1
                # ('lanczos2', False),
                # ('lanczos3', False),
            ],
            [0.25, 0.5, 1, 1.2, 1.5],
            [(0.99, 0.01), (0.8, 0.2), (0.9, 1.5), (467, 733)],
        )
    ),
)
def test_drizzle_weights_squared_pscale(kernel_fc, pscale, weights):
    n = 25
    shape = (n, n)

    # unpack parameters:
    kernel, fc = kernel_fc

    # pixel values in input data:
    dataval = [1.0, 7.0]

    # pixel values in input variance:
    varval = [0.5, 50]

    # input coordinate grid:
    y, x = np.indices(shape, dtype=np.float64)
    pixmap = np.dstack([x, y]) / pscale

    data = [np.zeros(shape, dtype=np.float32) for _ in range(2)]
    weight = [np.zeros(shape, dtype=np.float32) for _ in range(2)]
    var = [np.zeros(shape, dtype=np.float32) for _ in range(2)]

    xc = yc = n // 2
    sl = np.s_[yc - 4 : yc + 5, xc - 4 : xc + 5]
    for k in range(2):
        data[k][sl] = dataval[k]
        weight[k][sl] = weights[k]
        var[k][sl] = varval[k]

    out_shape = (int(pixmap[..., 1].max()) + 1, int(pixmap[..., 0].max()) + 1)

    # create a Drizzle object
    driz = resample.Drizzle(
        kernel=kernel,
        out_shape=out_shape,
        fillval=0.0,
        fillval2=0.0,
    )

    # resample & add input images
    for k in range(2):
        driz.add_image(
            data=data[k],
            exptime=1.0,
            pixmap=pixmap,
            weight_map=weight[k],
            data2=var[k],
        )

    mask = driz.out_ctx[0] > 0
    n_nonzero = np.sum(data[0] > 0.0)

    rtol = 1.0e-6 if fc else 0.15

    ideal_output = np.dot(dataval, weights) * n_nonzero
    ideal_output2 = np.dot(varval, np.square(weights)) / np.sum(weights) ** 2

    tflux = np.sum(driz.out_img[mask] * driz.out_wht[mask])
    tflux2 = np.max(driz.out_img2[0])

    # check output flux:
    assert np.allclose(tflux, ideal_output, rtol=rtol, atol=0.0)

    # check output variance:
    # less restrictive (to account for pixel overlap variations):
    assert np.max(tflux2) <= ideal_output2 * (1 + rtol) and np.max(
        tflux2
    ) >= 0.25 * ideal_output2 * (1 - rtol)

    # more restrictive check assumes pixels have good overlaps:
    assert np.allclose(tflux2, ideal_output2, rtol=rtol, atol=0.0)


def test_drizzle_weights_squared_bad_inputs():
    n = 21
    in_shape = (n, n)
    kernel = "square"

    # input coordinate grid:
    y, x = np.indices(in_shape, dtype=np.float64)

    in_sci1 = np.zeros(in_shape, dtype=np.float32)
    in_wht1 = np.zeros(in_shape, dtype=np.float32)
    in_sci1_sq = np.zeros(in_shape, dtype=np.float32)

    in_sci2 = np.zeros(in_shape, dtype=np.float32)
    in_wht2 = np.zeros(in_shape, dtype=np.float32)
    in_sci2_sq = np.zeros(in_shape, dtype=np.float32)

    pixmap = np.dstack([x, y])

    out_shape = (int(y.max()) + 1, int(x.max()) + 1)

    out_img = np.zeros(out_shape, dtype=np.float32)
    out_img2 = np.zeros(out_shape, dtype=np.float32)
    out_img2b = np.zeros(out_shape, dtype=np.float32)

    # 1 - test same number of data2 is used each time:
    driz = resample.Drizzle(
        kernel=kernel,
    )

    assert driz.out_img is None
    assert driz.out_img2 is None
    assert driz.total_exptime == 0.0

    driz.add_image(
        data=in_sci1,
        exptime=1.0,
        pixmap=pixmap,
        weight_map=in_wht1,
        data2=None,
    )
    assert driz.out_img2 is None

    with pytest.raises(ValueError) as err_info:
        driz.add_image(
            data=in_sci2,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht2,
            data2=in_sci2_sq,
        )
    assert str(err_info.value).startswith("Mismatch between the number of 'out_img2' images")

    # 2 - test same number of data2 is used each time:
    driz = resample.Drizzle(
        kernel=kernel,
    )

    driz.add_image(
        data=in_sci1,
        exptime=1.0,
        pixmap=pixmap,
        weight_map=in_wht1,
        data2=in_sci1_sq,
    )

    with pytest.raises(ValueError) as err_info:
        driz.add_image(
            data=in_sci2,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht2,
            data2=None,
        )
    assert str(err_info.value).startswith("Mismatch between the number of 'out_img2' images")

    # 3 - test same number of data2 is used each time:
    driz = resample.Drizzle(
        kernel=kernel,
        out_img2=[out_img2, out_img2b],
    )

    with pytest.raises(ValueError) as err_info:
        driz.add_image(
            data=in_sci2,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht2,
            data2=in_sci1_sq,
        )
    assert str(err_info.value).startswith("Mismatch between the number of 'out_img2' images")

    # 4 - test same number of data2 is used each time:
    driz = resample.Drizzle(
        kernel=kernel,
        out_img=out_img,
        out_img2=out_img2,
    )

    with pytest.raises(ValueError) as err_info:
        driz.add_image(
            data=in_sci2,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht2,
            data2=None,
        )
    assert str(err_info.value).startswith("Mismatch between the number of 'out_img2' images")

    # 5 - test mismatch between output data image and output variance image:
    out_img2 = np.zeros(tuple(s + 1 for s in out_shape), dtype=np.float32)

    with pytest.raises(ValueError) as err_info:
        driz = resample.Drizzle(
            kernel=kernel,
            out_img=out_img,
            out_img2=out_img2,
        )
    assert str(err_info.value).startswith("Inconsistent data shapes specified:")


def test_drizzle_weights_squared_array_shape_mismatch():
    n = 20
    in_shape = (n, n)
    in_shape1 = (n + 1, n + 1)
    kernel = "square"

    # input coordinate grid:
    y, x = np.indices(in_shape, dtype=np.float64)

    in_sci1 = np.zeros(in_shape, dtype=np.float32)
    in_sci1[n // 2, n // 2] = 2.222222222222
    in_sci1_sq = np.zeros(in_shape, dtype=np.float32)

    in_wht2 = np.zeros(in_shape1, dtype=np.float32)
    in_sci2_sq = np.zeros(in_shape1, dtype=np.float32)

    pixmap = np.dstack([x, y])

    out_shape = (int(y.max()) + 1, int(x.max()) + 1)
    out_shape1 = (out_shape[0] + 1, out_shape[1] + 1)

    out_img2 = np.zeros(out_shape, dtype=np.float32)
    out_img2b = np.zeros(out_shape1, dtype=np.float32)

    with pytest.raises(ValueError) as err_info:
        driz = resample.Drizzle(
            kernel=kernel,
            out_img2=[out_img2, out_img2b],
        )
    assert str(err_info.value).startswith("Inconsistent data shapes specified:")

    driz = resample.Drizzle(
        kernel=kernel,
        out_img=out_img2.copy(),
        out_img2=[out_img2, out_img2, None],
    )
    with pytest.raises(ValueError) as err_info:
        driz.add_image(
            data=in_sci1,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht2,
            data2=[in_sci1_sq, in_sci2_sq, None],
        )
    assert str(err_info.value).startswith("'data2' shape(s) is not consistent with 'data' shape.")

    driz = resample.Drizzle(
        kernel=kernel,
        out_img2=out_img2,
    )
    with pytest.raises(ValueError) as err_info:
        driz.add_image(
            data=in_sci1,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht2,
            data2=in_sci2_sq,
        )
    assert str(err_info.value).startswith("'data2' shape is not consistent with 'data' shape.")

    with pytest.raises(ValueError) as err_info:
        driz = resample.Drizzle(
            kernel=kernel,
            out_img2=[out_img2, out_img2b],
        )
    assert str(err_info.value).startswith("Inconsistent data shapes specified:")

    # wrong weight shape
    driz = resample.Drizzle(
        kernel=kernel,
    )
    with pytest.raises(ValueError) as err_info:
        driz.add_image(
            data=in_sci1,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht2,
        )
    assert str(err_info.value).startswith("'weight_map' shape is not consistent with 'data' shape.")

    # zero-sized variance array
    driz = resample.Drizzle(
        kernel=kernel, out_img2=[out_img2, out_img2.copy(), out_img2.copy(), None]
    )
    driz.add_image(
        data=in_sci1, exptime=1.0, pixmap=pixmap, data2=[in_sci1, in_sci1, np.array([]), None]
    )
    driz.add_image(data=in_sci1, exptime=1.0, pixmap=pixmap, data2=[in_sci1, None, in_sci1, None])
    assert np.allclose(np.nansum(driz.out_img2[0]), 2.0 * np.nansum(driz.out_img2[1]))
    assert np.allclose(np.nansum(driz.out_img2[0]), 2.0 * np.nansum(driz.out_img2[2]))
    assert np.allclose(0.0, np.nansum(driz.out_img2[3]))


@pytest.mark.parametrize(
    "kernel_fc, pscale_ratio, kscale_none",
    (
        x
        for x in product(
            [
                ("square", True),
                ("point", True),
                ("turbo", True),
                ("lanczos2", False),
                ("lanczos3", False),
                ("gaussian", False),
            ],
            [0.9, 1.0, 1.2],
            [False, True],
        )
    ),
)
def test_drizzle_var_identical_to_nonvar(kernel_fc, pscale_ratio, kscale_none):
    """Test that the resampled science image using code with support for
    variance-propagation is identical to the resampled science image
    using code without support for variance-propagation (original code).
    """
    kernel, fc = kernel_fc

    if kscale_none:
        kscale = None
    else:
        kscale = pscale_ratio

    amplitude = 100.0
    inwcs = wcs_from_file("j8bt06nyq_flt.fits", ext=1)
    insci = amplitude * np.random.random(inwcs.array_shape).astype(np.float32)
    inwht = np.ones_like(insci)
    output_wcs, _ = wcs_from_file("reference_square_image.fits", ext=1, return_data=True)

    pixmap = utils.calc_pixmap(
        inwcs,
        output_wcs,
    )

    driz1 = resample.Drizzle(
        kernel=kernel,
        fillval="NaN",
        out_shape=output_wcs.array_shape,
        exptime=0.0,
        begin_ctx_id=0,
        max_ctx_id=1,
        disable_ctx=False,
    )

    driz2 = resample.Drizzle(
        kernel=kernel,
        fillval="NaN",
        out_shape=output_wcs.array_shape,
        exptime=0.0,
        begin_ctx_id=0,
        max_ctx_id=1,
        disable_ctx=False,
    )

    if fc:
        driz1.add_image(
            insci,
            exptime=13.0,
            pixmap=pixmap,
            weight_map=inwht,
            iscale=pscale_ratio**2,
            pixel_scale_ratio=kscale,
            xmin=10,
            ymin=10,
            xmax=output_wcs.array_shape[0] - 10,
            ymax=output_wcs.array_shape[1] - 10,
        )
        driz2.add_image(
            insci,
            data2=insci,
            exptime=13.0,
            pixmap=pixmap,
            weight_map=inwht,
            iscale=pscale_ratio**2,
            pixel_scale_ratio=kscale,
            xmin=10,
            ymin=10,
            xmax=output_wcs.array_shape[0] - 10,
            ymax=output_wcs.array_shape[1] - 10,
        )
    else:
        with pytest.warns(Warning, match=f"Kernel '{kernel}' is not a flux-conserving kernel"):
            driz1.add_image(
                insci,
                exptime=13.0,
                pixmap=pixmap,
                weight_map=inwht,
                iscale=pscale_ratio**2,
                pixel_scale_ratio=kscale,
                xmin=10,
                ymin=10,
                xmax=output_wcs.array_shape[0] - 10,
                ymax=output_wcs.array_shape[1] - 10,
            )

        with pytest.warns(Warning, match=f"Kernel '{kernel}' is not a flux-conserving kernel"):
            driz2.add_image(
                insci,
                data2=insci,
                exptime=13.0,
                pixmap=pixmap,
                weight_map=inwht,
                iscale=pscale_ratio**2,
                pixel_scale_ratio=kscale,
                xmin=10,
                ymin=10,
                xmax=output_wcs.array_shape[0] - 10,
                ymax=output_wcs.array_shape[1] - 10,
            )

    assert np.allclose(
        driz1.out_img,
        driz2.out_img,
        rtol=0.0,
        atol=5.0 * amplitude * np.finfo(np.float32).eps,
        equal_nan=True,
    ), "Resampled science images are not identical."

    assert np.allclose(
        driz1.out_wht,
        driz2.out_wht,
        rtol=0.0,
        atol=5.0 * amplitude * np.finfo(np.float32).eps,
        equal_nan=True,
    ), "Resampled weight images are not identical."

    assert np.all(driz1.out_ctx == driz2.out_ctx), "Context images are not identical."


@pytest.mark.parametrize("create_out_dq", [True, False])
@pytest.mark.parametrize("shift", [0.0, 0.5])
@pytest.mark.parametrize("add_non_dq_image", [True, False])
def test_drizzle_dq_propagation(create_out_dq, shift, add_non_dq_image):
    n = 200
    in_shape = (n, n)

    # input coordinate grid:
    y, x = np.indices(in_shape, dtype=np.float64) + shift

    # simulate data:
    in_sci = np.ones(in_shape, dtype=np.float32)
    in_wht = np.ones(in_shape, dtype=np.float32)
    # use int16 to test up to 16 bits and signed integers
    in_dq1 = np.zeros(in_shape, dtype=np.int16)
    # use uint32 to test up to 32 bits and unsigned integers (default type)
    in_dq2 = np.zeros(in_shape, dtype=np.uint32)
    if create_out_dq:
        out_dq = np.zeros(in_shape, dtype=np.uint32)
    else:
        out_dq = None

    xyc = n // 2
    in_dq1[xyc, xyc] = 1 << 0
    in_dq1[xyc + 1, xyc] = 1 << 1
    in_dq1[xyc, xyc + 1] = 1 << 2
    in_dq1[xyc + 1, xyc + 1] = 1 << 3

    in_dq2[xyc, xyc] = 1 << 4
    in_dq2[xyc + 1, xyc] = 1 << 5
    in_dq2[xyc, xyc + 1] = 1 << 6
    in_dq2[xyc + 1, xyc + 1] = 1 << 7

    driz = resample.Drizzle(
        kernel="square",
        out_dq=out_dq,
    )

    pixmap = np.dstack([x, y])

    driz.add_image(
        in_sci,
        dq=in_dq1,
        exptime=1.0,
        pixmap=pixmap,
        weight_map=in_wht,
    )

    if shift == 0.0:
        assert driz.out_dq[xyc, xyc] == in_dq1[xyc, xyc]
        assert driz.out_dq[xyc + 1, xyc] == in_dq1[xyc + 1, xyc]
        assert driz.out_dq[xyc, xyc + 1] == in_dq1[xyc, xyc + 1]
        assert driz.out_dq[xyc + 1, xyc + 1] == in_dq1[xyc + 1, xyc + 1]
    else:
        # with shift=0.5 all 4 input pixels should contribute to the output
        # pixel at (xyc+1, xyc+1)
        assert driz.out_dq[xyc + 1, xyc + 1] == sum(1 << i for i in range(4))

    if add_non_dq_image:
        driz.add_image(
            in_sci,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht,
        )

        if shift == 0.0:
            assert driz.out_dq[xyc, xyc] == in_dq1[xyc, xyc]
            assert driz.out_dq[xyc + 1, xyc] == in_dq1[xyc + 1, xyc]
            assert driz.out_dq[xyc, xyc + 1] == in_dq1[xyc, xyc + 1]
            assert driz.out_dq[xyc + 1, xyc + 1] == in_dq1[xyc + 1, xyc + 1]
        else:
            # with shift=0.5 all 4 input pixels should contribute to the output
            # pixel at (xyc+1, xyc+1)
            assert driz.out_dq[xyc + 1, xyc + 1] == sum(1 << i for i in range(4))

    driz.add_image(
        in_sci,
        dq=in_dq2,
        exptime=1.0,
        pixmap=pixmap,
        weight_map=in_wht,
    )

    in_dq_total = in_dq1 + in_dq2

    if shift == 0.0:
        assert driz.out_dq[xyc, xyc] == in_dq_total[xyc, xyc]
        assert driz.out_dq[xyc + 1, xyc] == in_dq_total[xyc + 1, xyc]
        assert driz.out_dq[xyc, xyc + 1] == in_dq_total[xyc, xyc + 1]
        assert driz.out_dq[xyc + 1, xyc + 1] == in_dq_total[xyc + 1, xyc + 1]
    else:
        # with shift=0.5 all 4 input pixels should contribute to the output
        # pixel at (xyc+1, xyc+1)
        assert driz.out_dq[xyc + 1, xyc + 1] == sum(1 << i for i in range(8))

    if add_non_dq_image:
        driz.add_image(
            in_sci,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht,
        )

        if shift == 0.0:
            assert driz.out_dq[xyc, xyc] == in_dq_total[xyc, xyc]
            assert driz.out_dq[xyc + 1, xyc] == in_dq_total[xyc + 1, xyc]
            assert driz.out_dq[xyc, xyc + 1] == in_dq_total[xyc, xyc + 1]
            assert driz.out_dq[xyc + 1, xyc + 1] == in_dq_total[xyc + 1, xyc + 1]
        else:
            # with shift=0.5 all 4 input pixels should contribute to the output
            # pixel at (xyc+1, xyc+1)
            assert driz.out_dq[xyc + 1, xyc + 1] == sum(1 << i for i in range(8))


def test_drizzle_dq_propagation_wrong_shape():
    n = 200
    in_shape = (n, n)

    # input coordinate grid:
    y, x = np.indices(in_shape, dtype=np.float64)

    # simulate data:
    in_sci = np.ones(in_shape, dtype=np.float32)
    in_wht = np.ones(in_shape, dtype=np.float32)
    in_dq = np.zeros(tuple(i + 1 for i in in_shape), dtype=np.uint32)
    out_img = np.zeros(in_shape, dtype=np.float32)
    out_dq = np.zeros(tuple(i + 1 for i in in_shape), dtype=np.uint32)

    with pytest.raises(ValueError) as err_info:
        driz = resample.Drizzle(
            kernel="square",
            out_img=out_img,
            out_dq=out_dq,
        )
    assert str(err_info.value).startswith("Inconsistent data shapes specified:")

    driz = resample.Drizzle(
        kernel="square",
    )

    pixmap = np.dstack([x, y])

    with pytest.raises(ValueError) as err_info:
        driz.add_image(
            in_sci,
            dq=in_dq,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht,
        )
    assert str(err_info.value).startswith("'dq' shape is not consistent with 'data' shape.")


def test_drizzle_dq_propagation_wrong_type():
    n = 20
    in_shape = (n, n)

    # input coordinate grid:
    y, x = np.indices(in_shape, dtype=np.float64)

    # simulate data:
    in_sci = np.ones(in_shape, dtype=np.float32)
    in_wht = np.ones(in_shape, dtype=np.float32)
    in_dq = np.zeros(in_shape, dtype=np.uint64)
    out_img = np.zeros(in_shape, dtype=np.float32)
    out_dq = np.zeros(in_shape, dtype=np.uint64)

    with pytest.raises(TypeError) as err_info:
        driz = resample.Drizzle(
            kernel="square",
            out_img=out_img,
            out_dq=out_dq,
        )
    assert str(err_info.value).startswith(
        "'out_dq' must be of an unsigned integer type with itemsize of 4 bytes or less"
    )

    driz = resample.Drizzle(
        kernel="square",
    )

    pixmap = np.dstack([x, y])

    with pytest.raises(TypeError) as err_info:
        driz.add_image(
            in_sci,
            dq=in_dq,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=in_wht,
        )
    assert str(err_info.value).startswith(
        "'dq' must be of an unsigned integer type with itemsize of 4 bytes or less."
    )


@pytest.mark.parametrize(
    "kernel, pscale_ratio, use_var",
    (
        x
        for x in product(
            [
                "square",
                "point",
                "turbo",
                "lanczos2",
                "lanczos3",
                "gaussian",
            ],
            [0.9, 1.2, 0.3],
            [True, False],
        )
    ),
)
@pytest.mark.filterwarnings(r"ignore:Argument 'scale' has been deprecated.*:DeprecationWarning")
@pytest.mark.filterwarnings(r"ignore:Kernel '.*' is not a flux-conserving kernel:Warning")
def test_drizzle_ipscale_same_as_scale(kernel, pscale_ratio, use_var):
    """Test that the resampled science image using new "pixel_scale_ratio" and
    "iscale" parameters is identical to the resampled science image
    using the old "scale" parameter.

    TODO: remove this test when support for "scale" is removed.
    """
    amplitude = 100.0
    inwcs = wcs_from_file("j8bt06nyq_flt.fits", ext=1)
    insci = amplitude * np.random.random(inwcs.array_shape).astype(np.float32)
    inwht = np.ones_like(insci)
    output_wcs, _ = wcs_from_file("reference_square_image.fits", ext=1, return_data=True)

    pixmap = utils.calc_pixmap(
        inwcs,
        output_wcs,
    )

    driz1 = resample.Drizzle(
        kernel=kernel,
        fillval="NaN",
        out_shape=output_wcs.array_shape,
        exptime=0.0,
        begin_ctx_id=0,
        max_ctx_id=1,
        disable_ctx=False,
    )

    driz2 = resample.Drizzle(
        kernel=kernel,
        fillval="NaN",
        out_shape=output_wcs.array_shape,
        exptime=0.0,
        begin_ctx_id=0,
        max_ctx_id=1,
        disable_ctx=False,
    )

    driz1.add_image(
        insci,
        exptime=13.0,
        pixmap=pixmap,
        weight_map=inwht,
        iscale=pscale_ratio**2,
        pixel_scale_ratio=pscale_ratio,
        xmin=10,
        ymin=10,
        xmax=output_wcs.array_shape[0] - 10,
        ymax=output_wcs.array_shape[1] - 10,
    )

    if use_var:
        kwargs = {"data2": insci}
    else:
        kwargs = {}

    driz2.add_image(
        insci,
        exptime=13.0,
        pixmap=pixmap,
        weight_map=inwht,
        scale=pscale_ratio,
        xmin=10,
        ymin=10,
        xmax=output_wcs.array_shape[0] - 10,
        ymax=output_wcs.array_shape[1] - 10,
        **kwargs,
    )

    assert np.allclose(
        driz1.out_img,
        driz2.out_img,
        rtol=0.0,
        atol=5.0 * amplitude * np.finfo(np.float32).eps,
        equal_nan=True,
    ), "Resampled science images are not identical."

    assert np.allclose(
        driz1.out_wht,
        driz2.out_wht,
        rtol=0.0,
        atol=5.0 * amplitude * np.finfo(np.float32).eps,
        equal_nan=True,
    ), "Resampled weight images are not identical."

    assert np.all(driz1.out_ctx == driz2.out_ctx), "Context images are not identical."
