"""
The `drizzle` module defines the `Drizzle` class, for combining input
images into a single output image using the drizzle algorithm.
"""
import numpy as np

from . import cdrizzle


SUPPORTED_DRIZZLE_KERNELS = [
    "square",
    "gaussian",
    "point",
    "tophat",
    "turbo",
    "lanczos2",
    "lanczos3"
]

CTX_PLANE_BITS = 32


class Drizzle():
    def __init__(self, n_images=1, kernel="square",
                 fillval=None, out_shape=None, out_sci=None, out_wht=None,
                 out_ctx=None):
        """
        n_images : int, optional
            The number of images expected to be added to the resampled
            output. When it is a positive number and ``out_ctx`` is `None`,
            it allows to pre-allocate the necessary array for the output context
            image. If the actual number of input images that will be resampled
            will exceed initial allocation for the context image, additional
            context planes will be added as needed (context array will "grow"
            in the third dimention as new input images are added.)
            This parameter is ignored when ``out_ctx`` is provided.

        kernel: str, optional
            The name of the kernel used to combine the input. The choice of
            kernel controls the distribution of flux over the kernel. The kernel
            names are: "square", "gaussian", "point", "tophat", "turbo",
            "lanczos2", and "lanczos3". The square kernel is the default.

        out_shape : tuple, None, optional
            Shape (`numpy` order ``(Ny, Nx)``) of the output images (context
            image will have a third dimension of size proportional to the number
            of input images). This parameter is helpful when neither
            ``out_sci``, ``out_wht``, nor ``out_ctx`` images are provided.

        fillval: float, None, str, optional
            The value of output pixels that did not have contributions from
            input images' pixels. When ``fillval`` is either `None` or
            ``"INDEF"`` and ``out_sci`` is provided, the values of ``out_sci``
            will not be modified. When ``fillval`` is either `None` or
            ``"INDEF"`` and ``out_sci`` is **not provided**, the values of
            ``out_sci`` will be initialized to `numpy.nan`. If ``fillval``
            is a string that can be converted to a number, then the output
            pixels with no contributions from input images will be set to this
            ``fillval`` value.

        out_sci : 2d array of float32, optional
            A 2D numpy array containing the output image produced by
            drizzling. On the first call it should be set to zero.
            Subsequent calls it will hold the intermediate results

        out_wht : 2D array of float32, optional
            A 2D numpy array containing the output counts. On the first
            call it should be set to zero. On subsequent calls it will
            hold the intermediate results.

        out_ctx : 2D or 3D array of int32, optional
            A 2D or 3D numpy array holding a bitmap of which image was an input
            for each output pixel. Should be integer zero on first call.
            Subsequent calls hold intermediate results.

        """
        self.ctx_id = -1  # the ID of the *last* image to be resampled
        self.exptime = 0.0

        if kernel.lower() not in SUPPORTED_DRIZZLE_KERNELS:
            raise ValueError(f"Kernel '{kernel}' is not supported.")
        self.kernel = kernel

        if fillval is None:
            fillval = "INDEF"

        elif isinstance(str, fillval):
            fillval = fillval.strip()
            if fillval == "":
                fillval = "INDEF"
            elif fillval.upper() == "INDEF":
                pass
            else:
                float(fillval)
                fillval = str(fillval)

        if out_sci is None and fillval == "INDEF":
            fillval = "NaN"

        self.fillval = fillval

        if (out_shape is None and out_sci is None and out_wht is None and
                out_ctx is None):
            raise ValueError(
                "'out_shape' cannot be None when all output arrays are None."
            )

        shapes = set()

        if out_sci is not None:
            out_sci = np.asarray(out_sci, dtype=np.float32)
            shapes.add(out_sci.shape)

        if out_wht is not None:
            out_wht = np.asarray(out_wht, dtype=np.float32)
            shapes.add(out_wht.shape)

        if out_ctx is not None:
            out_ctx = np.asarray(out_ctx, dtype=np.int32)
            if len(out_ctx.shape) == 3:
                shapes.add(out_ctx.shape[1:])
            else:
                shapes.add(out_ctx.shape)

        if out_shape is None:
            if not shapes:
                raise ValueError(
                    "'out_shape' cannot be None when all output arrays are "
                    "None."
                )
            out_shape = shapes[0]
        else:
            shapes.add(out_shape)

        if len(shapes) == 1:
            self.out_shape = shapes.pop()
        elif len(shapes) > 1:
            raise ValueError(
                "Inconsistent data shapes specified: 'out_shape' and/or "
                "out_sci, out_wht, out_ctx have different shapes."
            )
        else:
            raise ValueError(
                "Either 'out_shape' and/or out_sci, out_wht, out_ctx must be "
                "provided."
            )

        self._alloc_output_arrays(
            out_shape=self.out_shape,
            n_images=n_images,
            out_sci=out_sci,
            out_wht=out_wht,
            out_ctx=out_ctx,
        )

    def _alloc_output_arrays(self, out_shape, n_images, out_sci, out_wht,
                             out_ctx):
        # allocate arrays as needed:
        if out_sci is None:
            self.out_sci = np.empty(out_shape, dtype=np.float32)
            self.out_sci.fill(self.fillval)

        if out_wht is None:
            self.out_wht = np.zeros(out_shape, dtype=np.float32)

        if out_ctx is None:
            n_ctx_planes = (n_images - 1) // CTX_PLANE_BITS + 1
            if n_ctx_planes == 1:
                ctx_shape = out_shape
            else:
                ctx_shape = (n_ctx_planes, ) + out_shape
            self.out_ctx = np.zeros(ctx_shape, dtype=np.int32)

    def _increment_ctx_id(self):
        self.ctx_id += 1
        if self.ctx_id < 0:
            ValueError("Invalid context image ID")

        self._plane_no = self.ctx_id // CTX_PLANE_BITS
        depth = 1 if len(self.out_ctx.shape) == 2 else self.out_ctx.shape[0]

        if self._plane_no >= depth:
            # Add a new plane to the context image if planeid overflows
            plane = np.zeros_like(self.out_shape, np.int32)
            self.outcon = np.append(self.out_ctx, [plane], axis=0)

        return (self._plane_no, self.ctx_id % CTX_PLANE_BITS)

    def add_image(self, data, exptime, pixmap, scale=1.0,
                  weight_map=None, wht_scale=1.0, pixfrac=1.0, in_units='cps',
                  xmin=None, xmax=None, ymin=None, ymax=None):
        """

        Resample and add an image to the cumulative output image. Also, update
        output total weight image and context images.

        TODO: significantly expand this with examples, especially for
        exptime, expsq, and ivm weightings.

        Parameters
        ----------

        data : 2d array
            A 2d numpy array containing the input image to be drizzled.
            it is an error to not supply an image.

        exptime : float
            The exposure time of the input image, a positive number. The
            exposure time is used to scale the image if the units are counts.

        pixmap : array
            TODO: add description.

        scale : float, optional
            The pixel scale of the input image. Conceptually, this is the
            linear dimension of a side of a pixel in the input image, but it
            is not limited to this and can be set to change how the drizzling
            algorithm operates.

        weight_map : 2d array
            A 2d numpy array containing the pixel by pixel weighting.
            Must have the same dimensions as ``data``.

            TODO: I think this is wrong: If none is supplied, the weghting is set to one.

        wht_scale : float
            A scaling factor applied to the pixel by pixel weighting.

        pixfrac : float, optional
            The fraction of a pixel that the pixel flux is confined to. The
            default value of 1 has the pixel flux evenly spread across the image.
            A value of 0.5 confines it to half a pixel in the linear dimension,
            so the flux is confined to a quarter of the pixel area when the square
            kernel is used.

        in_units : str
            The units of the input image. The units can either be "counts"
            or "cps" (counts per second.)

        xmin : float, optional
            This and the following three parameters set a bounding rectangle
            on the input image. Only pixels on the input image inside this
            rectangle will have their flux added to the output image. Xmin
            sets the minimum value of the x dimension. The x dimension is the
            dimension that varies quickest on the image. If the value is zero,
            no minimum will be set in the x dimension. All four parameters are
            zero based, counting starts at zero.

        xmax : float, optional
            Sets the maximum value of the x dimension on the bounding box
            of the input image. If the value is zero, no maximum will
            be set in the x dimension, the full x dimension of the output
            image is the bounding box.

        ymin : float, optional
            Sets the minimum value in the y dimension on the bounding box. The
            y dimension varies less rapidly than the x and represents the line
            index on the input image. If the value is zero, no minimum  will be
            set in the y dimension.

        ymax : float, optional
            Sets the maximum value in the y dimension. If the value is zero, no
            maximum will be set in the y dimension,  the full x dimension
            of the output image is the bounding box.

        """
        plane_no, id_in_plane = self._increment_ctx_id()

        # Ensure that the fillval parameter gets properly interpreted
        # for use with tdriz
        if in_units == 'cps':
            expscale = 1.0
        else:
            expscale = exptime

        self.exptime += exptime

        data = np.asarray(data, dtype=np.float32)
        pixmap = np.asarray(pixmap, dtype=np.float64)

        if pixmap.shape[:2] != data.shape:
            raise ValueError(
                "'pixmap' shape is not consistent with 'data' shape."
            )

        # TODO: this is code that would enable initializer to not need output
        #       image shape at all and set output image shape based on
        #       output coordinates from the pixmap. If this is enabled, then
        #       some checks in __init__ may be removed:
        # if self.out_shape is None:
        #     pmap_xmin = int(np.floor(np.min(pixmap[:, :, 0])))
        #     pmap_xmax = int(np.ceil(np.max(pixmap[:, :, 0])))
        #     pmap_ymin = int(np.floor(np.min(pixmap[:, :, 1])))
        #     pmap_ymax = int(np.ceil(np.max(pixmap[:, :, 1])))
        #     pixmap = pixmap.copy()
        #     pixmap[:, :, 0] -= pmap_xmin
        #     pixmap[:, :, 1] -= pmap_ymin
        #     self.out_shape = (
        #         pmap_xmax - pmap_xmin + 1,
        #         pmap_ymax - pmap_ymin + 1
        #     )

        if xmin is None or xmin < 0:
            xmin = 0

        if ymin is None or ymin < 0:
            ymin = 0

        if xmax is None or xmax > self.out_shape[1] - 1:
            xmax = self.out_shape[1] - 1

        if ymax is None or ymax > self.out_shape[0] - 1:
            ymax = self.out_shape[0] - 1

        if weight_map is not None:
            weight_map = np.asarray(weight_map, dtype=np.float32)
        else:  # TODO: this should not be needed after C code modifications
            weight_map = np.ones_like(data)

        pixmap = np.asarray(pixmap, dtype=np.float64)

        # TODO: probably tdriz should be modified to not return version.
        #       we should not have git, Python, C, ... versions

        # TODO: While drizzle code in cdrizzlebox.c supports weight_map=None,
        #       cdrizzleapi.c does not. It should be modified to support this
        #       for performance reasons.
        if len(self.out_ctx.shape) == 2:
            assert plane_no == 0
            ctx_plane = self.out_ctx
        else:
            ctx_plane = self.out_ctx[plane_no]

        _vers, nmiss, nskip = cdrizzle.tdriz(
            data,
            weight_map,
            pixmap,
            self.out_sci,
            self.out_wht,
            ctx_plane,
            uniqid=id_in_plane + 1,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            scale=scale, # scales image intensity. usually equal to pixel scale
            pixfrac=pixfrac,
            kernel=self.kernel,
            in_units=in_units,
            expscale=expscale,
            wtscale=wht_scale,
            fillstr=self.fillval
        )
        self._cversion = _vers  # TODO: probably not needed

        return nmiss, nskip


def blot_image(data, pixmap, pix_ratio, exptime, output_pixel_shape,
               interp='poly5', sinscl=1.0):
    """
    Resample input image using interpolation to an output grid.
    Typically, this is used to resample the resampled image that is the
    result of multiple applications of ``Drizzle.add_image`` **if it is well
    sampled** back to the pixel coordinate system of input (with distorted
    WCS) images of ``Drizzle.add_image``.

    Parameters
    ----------

    data : 2D array
        Input numpy array of the source image in units of 'cps'.

    pixmap : 3D array
        The world coordinate system to resample on.

    output_pixel_shape : tuple of int
        A tuple of two integer numbers indicating the dimensions of the output
        image ``(Nx, Ny)``.

    pix_ratio : float
        Ratio of the input image pixel scale to the ouput image pixel scale.

    exptime : float
        The exposure time of the input image.

    interp : str, optional
        The type of interpolation used in the resampling. The
        possible values are "nearest" (nearest neighbor interpolation),
        "linear" (bilinear interpolation), "poly3" (cubic polynomial
        interpolation), "poly5" (quintic polynomial interpolation),
        "sinc" (sinc interpolation), "lan3" (3rd order Lanczos
        interpolation), and "lan5" (5th order Lanczos interpolation).

    sincscl : float, optional
        The scaling factor for sinc interpolation.
    """

    out_sci = np.zeros(output_pixel_shape[::-1], dtype=np.float32)

    cdrizzle.tblot(data, pixmap, out_sci, scale=pix_ratio, kscale=1.0,
                   interp=interp, exptime=exptime, misval=0.0, sinscl=sinscl)

    return out_sci
