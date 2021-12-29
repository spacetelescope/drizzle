import math
import os
import pytest

import numpy as np
from astropy import wcs
from astropy.io import fits

from drizzle import drizzle, cdrizzle


TEST_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TEST_DIR, 'data')


ok = False


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

    center = [0.0, 0.0, 0.0]
    for y in range(ylo, yhi):
        for x in range(xlo, xhi):
            center[0] += y * image[y,x]
            center[1] += x * image[y,x]
            center[2] += image[y,x]

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
        if (abs(list_of_centroids[i][0] - point[0]) < int(size / 2) and
                abs(list_of_centroids[i][1] - point[1]) < int(size / 2)):
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
    for center2 in list_of_centroids:
        center1 = centroid(image1, size, center2)
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
    fd = open(fname, 'w')
    fd.write("*** %s ***\n" % title)

    if len(distances) == 0:
        diff = [0.0, 0.0, 0.0]
        fd.write("No matches!!\n")

    elif len(distances) == 1:
        diff = [distances[0][0], distances[0][0], distances[0][0]]

        fd.write("1 match\n")
        fd.write("distance = %f flux difference = %f\n" %
                 (distances[0][0], distances[0][1]))

        for j in range(2, 4):
            ylo = int(distances[0][j][0]) - 1
            yhi = int(distances[0][j][0]) + 2
            xlo = int(distances[0][j][1]) - 1
            xhi = int(distances[0][j][1]) + 2
            subimage = images[j][ylo:yhi,xlo:xhi]
            fd.write("\n%s image centroid = (%f,%f) image flux = %f\n" %
                     (im_type[j], distances[0][j][0], distances[0][j][1],
                      distances[0][j][2]))
            fd.write(str(subimage) + "\n")

    else:
        fd.write("%d matches\n" % len(distances))

        for k in range(0,3):
            i = indexes[k]
            diff.append(distances[i][0])
            fd.write("\n%s distance = %f flux difference = %f\n" %
                     (stats[k],distances[i][0], distances[i][1]))

            for j in range(2, 4):
                ylo = int(distances[i][j][0]) - 1
                yhi = int(distances[i][j][0]) + 2
                xlo = int(distances[i][j][1]) - 1
                xhi = int(distances[i][j][1]) + 2
                subimage = images[j][ylo:yhi,xlo:xhi]
                fd.write("\n%s %s image centroid = (%f,%f) image flux = %f\n" %
                         (stats[k], im_type[j], distances[i][j][0],
                          distances[i][j][1], distances[i][j][2]))
                fd.write(str(subimage) + "\n")

    fd.close()
    return tuple(diff)


def make_point_image(input_image, point, value):
    """
    Create an image with a single point set
    """
    output_image = np.zeros(input_image.shape, dtype=input_image.dtype)
    output_image[point] = value
    return output_image


def make_grid_image(input_image, spacing, value):
    """
    Create an image with points on a grid set
    """
    output_image = np.zeros(input_image.shape, dtype=input_image.dtype)

    shape = output_image.shape
    half_space = int(spacing / 2)
    for y in range(half_space, shape[0], spacing):
        for x in range(half_space, shape[1], spacing):
            output_image[y,x] = value

    return output_image


def print_wcs(title, wcs):
    """
    Print the wcs header cards
    """
    print("=== %s ===" % title)
    print(wcs.to_header_string())


def read_image(filename):
    """
    Read the image from a fits file
    """
    path = os.path.join(DATA_DIR, filename)
    hdu = fits.open(path)

    image = hdu[1].data
    hdu.close()
    return image


def read_wcs(filename):
    """
    Read the wcs of a fits file
    """
    path = os.path.join(DATA_DIR, filename)
    hdu = fits.open(path)
    the_wcs = wcs.WCS(hdu[1].header)
    hdu.close()
    return the_wcs


def test_square_with_point(tmpdir):
    """
    Test do_driz square kernel with point
    """
    output = str(tmpdir.join('output_square_point.fits'))
    output_difference = str(tmpdir.join('difference_square_point.txt'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')
    output_template = os.path.join(DATA_DIR, 'reference_square_point.fits')

    insci = read_image(input_file)
    inwcs = read_wcs(input_file)
    insci = make_point_image(insci, (500, 200), 100.0)
    inwht = np.ones(insci.shape,dtype=insci.dtype)
    output_wcs = read_wcs(output_template)

    driz = drizzle.Drizzle(outwcs=output_wcs, wt_scl="")
    driz.add_image(insci, inwcs, inwht=inwht)

    if ok:
        driz.write(output_template)
    else:
        driz.write(output)
        template_data = read_image(output_template)

        min_diff, med_diff, max_diff = centroid_statistics("square with point",
                                                           output_difference,
                                                           driz.outsci,
                                                           template_data, 20.0, 8)

        assert med_diff < 1.0e-6
        assert max_diff < 1.0e-5


@pytest.mark.parametrize(
    'kernel', ['square', 'point', 'turbo', 'gaussian', 'lanczos3']
)
def test_zero_input_weight(kernel):
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
    cdrizzle.tdriz(
        insci, inwht, pixmap,
        outsci, outwht, outctx,
        uniqid=1,
        xmin=0, xmax=400,
        ymin=0, ymax=200,
        pixfrac=1,
        kernel=kernel,
        in_units='cps',
        expscale=1,
        wtscale=1,
        fillstr='INDEF'
    )

    # check that no pixel with 0 weight has any counts:
    assert np.sum(np.abs(outsci[(outwht == 0)])) == 0.0


def test_square_with_grid(tmpdir):
    """
    Test do_driz square kernel with grid
    """
    output = str(tmpdir.join('output_square_grid.fits'))
    output_difference = str(tmpdir.join('difference_square_grid.txt'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')
    output_template = os.path.join(DATA_DIR, 'reference_square_grid.fits')

    insci = read_image(input_file)
    inwcs = read_wcs(input_file)
    insci = make_grid_image(insci, 64, 100.0)
    inwht = np.ones(insci.shape,dtype=insci.dtype)
    output_wcs = read_wcs(output_template)

    driz = drizzle.Drizzle(outwcs=output_wcs, wt_scl="")
    driz.add_image(insci, inwcs, inwht=inwht)

    if ok:
        driz.write(output_template)
    else:
        driz.write(output)
        template_data = read_image(output_template)

        min_diff, med_diff, max_diff = centroid_statistics("square with grid",
                                                           output_difference,
                                                           driz.outsci,
                                                           template_data, 20.0, 8)
        assert med_diff < 1.0e-6
        assert max_diff < 1.0e-5


def test_turbo_with_grid(tmpdir):
    """
    Test do_driz turbo kernel with grid
    """
    output = str(tmpdir.join('output_turbo_grid.fits'))
    output_difference = str(tmpdir.join('difference_turbo_grid.txt'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')
    output_template = os.path.join(DATA_DIR, 'reference_turbo_grid.fits')

    insci = read_image(input_file)
    inwcs = read_wcs(input_file)
    insci = make_grid_image(insci, 64, 100.0)
    inwht = np.ones(insci.shape,dtype=insci.dtype)
    output_wcs = read_wcs(output_template)

    driz = drizzle.Drizzle(outwcs=output_wcs, wt_scl="", kernel='turbo')
    driz.add_image(insci, inwcs, inwht=inwht)

    if ok:
        driz.write(output_template)
    else:
        driz.write(output)
        template_data = read_image(output_template)

        min_diff, med_diff, max_diff = centroid_statistics("turbo with grid",
                                                           output_difference,
                                                           driz.outsci,
                                                           template_data, 20.0, 8)

        assert med_diff < 1.0e-6
        assert max_diff < 1.0e-5


def test_gaussian_with_grid(tmpdir):
    """
    Test do_driz gaussian kernel with grid
    """
    output = str(tmpdir.join('output_gaussian_grid.fits'))
    output_difference = str(tmpdir.join('difference_gaussian_grid.txt'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')
    output_template = os.path.join(DATA_DIR, 'reference_gaussian_grid.fits')

    insci = read_image(input_file)
    inwcs = read_wcs(input_file)
    insci = make_grid_image(insci, 64, 100.0)
    inwht = np.ones(insci.shape,dtype=insci.dtype)
    output_wcs = read_wcs(output_template)

    driz = drizzle.Drizzle(outwcs=output_wcs, wt_scl="", kernel='gaussian')
    driz.add_image(insci, inwcs, inwht=inwht)

    if ok:
        driz.write(output_template)
    else:
        driz.write(output)
        template_data = read_image(output_template)

        min_diff, med_diff, max_diff = centroid_statistics("gaussian with grid",
                                                           output_difference,
                                                           driz.outsci,
                                                           template_data, 20.0, 8)

        assert med_diff < 1.0e-6
        assert max_diff < 2.0e-5


def test_lanczos_with_grid(tmpdir):
    """
    Test do_driz lanczos kernel with grid
    """
    output = str(tmpdir.join('output_lanczos_grid.fits'))
    output_difference = str(tmpdir.join('difference_lanczos_grid.txt'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')
    output_template = os.path.join(DATA_DIR, 'reference_lanczos_grid.fits')

    insci = read_image(input_file)
    inwcs = read_wcs(input_file)
    insci = make_grid_image(insci, 64, 100.0)
    inwht = np.ones(insci.shape,dtype=insci.dtype)
    output_wcs = read_wcs(output_template)

    driz = drizzle.Drizzle(outwcs=output_wcs, wt_scl="", kernel='lanczos3')
    driz.add_image(insci, inwcs, inwht=inwht)

    if ok:
        driz.write(output_template)
    else:
        driz.write(output)
        template_data = read_image(output_template)

        min_diff, med_diff, max_diff = centroid_statistics("lanczos with grid",
                                                           output_difference,
                                                           driz.outsci,
                                                           template_data, 20.0, 8)
        assert med_diff < 1.0e-6
        assert max_diff < 1.0e-5


def test_tophat_with_grid(tmpdir):
    """
    Test do_driz tophat kernel with grid
    """
    output = str(tmpdir.join('output_tophat_grid.fits'))
    output_difference = str(tmpdir.join('difference_tophat_grid.txt'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')
    output_template = os.path.join(DATA_DIR, 'reference_tophat_grid.fits')

    insci = read_image(input_file)
    inwcs = read_wcs(input_file)
    insci = make_grid_image(insci, 64, 100.0)
    inwht = np.ones(insci.shape,dtype=insci.dtype)
    output_wcs = read_wcs(output_template)

    driz = drizzle.Drizzle(outwcs=output_wcs, wt_scl="", kernel='tophat')
    driz.add_image(insci, inwcs, inwht=inwht)

    if ok:
        driz.write(output_template)
    else:
        driz.write(output)
        template_data = read_image(output_template)

        min_diff, med_diff, max_diff = centroid_statistics("tophat with grid",
                                                           output_difference,
                                                           driz.outsci,
                                                           template_data, 20.0, 8)
        assert med_diff < 1.0e-6
        assert max_diff < 1.0e-5


def test_point_with_grid(tmpdir):
    """
    Test do_driz point kernel with grid
    """
    output = str(tmpdir.join('output_point_grid.fits'))
    output_difference = str(tmpdir.join('difference_point_grid.txt'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')
    output_template = os.path.join(DATA_DIR, 'reference_point_grid.fits')

    insci = read_image(input_file)
    inwcs = read_wcs(input_file)
    insci = make_grid_image(insci, 64, 100.0)
    inwht = np.ones(insci.shape,dtype=insci.dtype)
    output_wcs = read_wcs(output_template)

    driz = drizzle.Drizzle(outwcs=output_wcs, wt_scl="", kernel='point')
    driz.add_image(insci, inwcs, inwht=inwht)

    if ok:
        driz.write(output_template)
    else:
        driz.write(output)
        template_data = read_image(output_template)

        min_diff, med_diff, max_diff = centroid_statistics("point with grid",
                                                           output_difference,
                                                           driz.outsci,
                                                           template_data, 20.0, 8)
        assert med_diff < 1.0e-6
        assert max_diff < 1.0e-5


def test_blot_with_point(tmpdir):
    """
    Test do_blot with point image
    """
    output = str(tmpdir.join('output_blot_point.fits'))
    output_difference = str(tmpdir.join('difference_blot_point.txt'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')
    output_template = os.path.join(DATA_DIR, 'reference_blot_point.fits')

    outsci = read_image(input_file)
    outwcs = read_wcs(input_file)
    outsci = make_point_image(outsci, (500, 200), 40.0)
    inwcs = read_wcs(output_template)

    driz = drizzle.Drizzle(outwcs=outwcs)
    driz.outsci = outsci

    driz.blot_image(inwcs)

    if ok:
        driz.write(output_template)
    else:
        driz.write(output)
        template_data = read_image(output_template)

        min_diff, med_diff, max_diff = centroid_statistics("blot with point",
                                                           output_difference,
                                                           driz.outsci,
                                                           template_data, 20.0, 16)
        assert med_diff < 1.0e-6
        assert max_diff < 1.0e-5


def test_blot_with_default(tmpdir):
    """
    Test do_blot with default grid image
    """
    output = str(tmpdir.join('output_blot_default.fits'))
    output_difference = str(tmpdir.join('difference_blot_default.txt'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')
    output_template = os.path.join(DATA_DIR, 'reference_blot_default.fits')

    outsci = read_image(input_file)
    outsci = make_grid_image(outsci, 64, 100.0)
    outwcs = read_wcs(input_file)
    inwcs = read_wcs(output_template)

    driz = drizzle.Drizzle(outwcs=outwcs)
    driz.outsci = outsci

    driz.blot_image(inwcs)

    if ok:
        driz.write(output_template)
    else:
        driz.write(output)
        template_data = read_image(output_template)

        min_diff, med_diff, max_diff = centroid_statistics("blot with defaults",
                                                           output_difference,
                                                           driz.outsci,
                                                           template_data, 20.0, 16)

        assert med_diff < 1.0e-6
        assert max_diff < 1.0e-5


def test_blot_with_lan3(tmpdir):
    """
    Test do_blot with lan3 grid image
    """
    output = str(tmpdir.join('output_blot_lan3.fits'))
    output_difference = str(tmpdir.join('difference_blot_lan3.txt'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')
    output_template = os.path.join(DATA_DIR, 'reference_blot_lan3.fits')

    outsci = read_image(input_file)
    outsci = make_grid_image(outsci, 64, 100.0)
    outwcs = read_wcs(input_file)
    inwcs = read_wcs(output_template)

    driz = drizzle.Drizzle(outwcs=outwcs)
    driz.outsci = outsci

    driz.blot_image(inwcs, interp="lan3")

    if ok:
        driz.write(output_template)
    else:
        driz.write(output)
        template_data = read_image(output_template)

        min_diff, med_diff, max_diff = centroid_statistics("blot with lan3",
                                                           output_difference,
                                                           driz.outsci,
                                                           template_data, 20.0, 16)
        assert med_diff < 1.0e-6
        assert max_diff < 1.0e-5


def test_blot_with_lan5(tmpdir):
    """
    Test do_blot with lan5 grid image
    """
    output = str(tmpdir.join('output_blot_lan5.fits'))
    output_difference = str(tmpdir.join('difference_blot_lan5.txt'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')
    output_template = os.path.join(DATA_DIR, 'reference_blot_lan5.fits')

    outsci = read_image(input_file)
    outsci = make_grid_image(outsci, 64, 100.0)
    outwcs = read_wcs(input_file)
    inwcs = read_wcs(output_template)

    driz = drizzle.Drizzle(outwcs=outwcs)
    driz.outsci = outsci

    driz.blot_image(inwcs, interp="lan5")

    if ok:
        driz.write(output_template)
    else:
        driz.write(output)
        template_data = read_image(output_template)

        min_diff, med_diff, max_diff = centroid_statistics("blot with lan5",
                                                           output_difference,
                                                           driz.outsci,
                                                           template_data, 20.0, 16)
        assert med_diff < 1.0e-6
        assert max_diff < 1.0e-5


def test_context_planes():
    """Reproduce error seen in issue #50"""
    shape = [10, 10]
    outwcs = wcs.WCS()
    outwcs.pixel_shape = shape
    driz = drizzle.Drizzle(outwcs=outwcs)

    image = np.ones(shape)
    inwcs = wcs.WCS()
    inwcs.pixel_shape = shape

    for i in range(32):
        driz.add_image(image, inwcs)
    assert driz.outcon.shape == (1, 10, 10)

    driz.add_image(image, inwcs)
    assert driz.outcon.shape == (2, 10, 10)
