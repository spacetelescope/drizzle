"""
The `drizzle` module defines the `Drizzle` class, for combining input
images into a single output image using the drizzle algorithm.
"""

import warnings

import numpy as np

from drizzle import cdrizzle

__all__ = ["Drizzle", "blot_image"]

SUPPORTED_DRIZZLE_KERNELS = [
    "square",
    "gaussian",
    "point",
    "turbo",
    "lanczos2",
    "lanczos3",
]

CTX_PLANE_BITS = 32


_DEPRECATED_ARG = object()


class Drizzle:
    """
    A class for managing resampling and co-adding of multiple images onto a
    common output grid. The main method of this class is :py:meth:`add_image`.
    The main functionality of this class is to resample and co-add multiple
    images onto one output image using the "drizzle" algorithm described in
    `Fruchter and Hook, PASP 2002 <https://doi.org/10.1086/338393>`_.
    In the simplest terms, it redistributes flux
    from input pixels to one or more output pixels based on the chosen kernel,
    supplied weights, and input-to-output coordinate transformations as defined
    by the ``pixmap`` argument. For more details, see :ref:`main-user-doc`.

    This class keeps track of the total exposure time of all co-added images
    and also of which input images have contributed to an output (resampled)
    pixel. This is accomplished via *context image*.

    Main outputs of :py:meth:`add_image` can be accessed as class properties
    ``out_img``, ``out_img2``, ``out_wht``, ``out_ctx``, and ``exptime``.

    .. warning::
        Output arrays (``out_img``, ``out_img2``, ``out_wht``, and ``out_ctx``)
        can be pre-allocated by the caller and be passed to the initializer or
        the class initializer can allocate these arrays based on other input
        parameters such as ``output_shape``. If caller-supplied output arrays
        have the correct type (`numpy.float32` for ``out_img``, ``out_img2``
        and ``out_wht``, `numpy.int32` for the ``out_ctx`` array and
        `numpy.uint32` for the ``out_dq`` array) and if
        ``out_ctx`` is large enough not to need to be resized, these arrays
        will be used as is and may be modified by the :py:meth:`add_image`
        method. If not, a copy of these arrays will be made when converting
        to the expected type (or expanding the context array).

    Scaling of input image data
    ---------------------------

    It is important to highlight that the drizzle algorithm computes
    *weighted mean* of input pixel values -- see equations (4) and (5) in
    `Fruchter and Hook, PASP 2002 <https://doi.org/10.1086/338393>`_.
    Therefore, it is important that all input pixel values that contribute
    to an output pixel are from the same distribution. In other words,
    input pixel values from different images must be on the same footing,
    i.e., they must be comparable and must be representative of the same
    physical quantity.

    For example, for Hubble Space Telescope data, calibrated images
    (i.e., ``*_flt.fits``, ``*_flc.fits``) are in unit of counts, counts per
    second, electrons, or electrons per second. To convert them to flux
    units (e.g., erg/cm^2/s/Angstrom), one needs to multiply these images
    by the ``PHOTFLAM``. Sometimes, images that are drizzle-combined have been
    observed at very different times (separated by many years) and the
    sensitivity of the instrument (represented by ``PHOTFLAM``) may have
    changed significantly. Other times a source is observed in different chips,
    i.e., the two chips of the Wide Field Camera. In such cases detector's
    sensitivity (``PHOTFLAM``) may be different for the images to be combined.
    Consequently, pixel values in these images may not be directly comparable
    and drizzle-combining such images would result in systematic errors.

    In this case, it is important to rescale images to the same flux units
    either by multiplying by the appropriate ``PHOTFLAM`` values or some
    other appropriate scaling factor before combining them using drizzle.
    This can be accomplished by using the ``iscale`` parameter of
    :py:meth:`add_image` which simply multiplies each input image by
    ``iscale``.

    Also, for the case of HST images that have flux units instead of surface
    brightness, if input images have different pixel scales, then the pixel
    values must be rescaled by the square of the pixel scale ratio (the linear
    dimension of a side of an output pixel as seen in the input image's
    coordinate frame) in order to preserve flux. In this case ``iscale`` is
    equivalent to ``s**2`` factor in equations (3) and (5) of
    `Fruchter and Hook, PASP 2002 <https://doi.org/10.1086/338393>`_
    (``s`` may be different for each input image).

    Output Science Image
    --------------------

    Output science image is obtained by computing *weighted mean* of input
    pixel values according to equations (4) and (5) in
    `Fruchter and Hook, PASP 2002 <https://doi.org/10.1086/338393>`_.
    The weights and coefficients in those equations will depend on the chosen
    kernel, input image weights, and pixel overlaps computed from ``pixmap``.

    Output Weight Image
    -------------------

    Output weight image stores the total weight of output science pixels
    according to equation (4) in
    `Fruchter and Hook, PASP 2002 <https://doi.org/10.1086/338393>`_.
    It depends on the chosen kernel, input image weights, and pixel overlaps
    computed from ``pixmap``.

    Output Context Image
    --------------------

    Each pixel in the context image is a bit field that encodes
    information about which input image has contributed to the corresponding
    pixel in the resampled data array. Context image uses 32 bit integers to
    encode this information and hence it can keep track of only 32 input images.
    The first bit corresponds to the first input image, the second bit
    corresponds to the second input image, and so on.
    We call this (0-indexed) order "context ID" which is represented by
    the ``ctx_id`` parameter/property. If the number of
    input images exceeds 32, then it is necessary to have multiple context
    images ("planes") to hold information about all input images, with the first
    plane encoding which of the first 32 images contributed to the output data
    pixel, the second plane representing next 32 input images (number 33-64),
    etc. For this reason, context array is either a 2D array (if the total
    number of resampled images is less than 33) of the type `numpy.int32` and
    shape ``(ny, nx)`` or a a 3D array of shape ``(np, ny, nx)`` where ``nx``
    and ``ny`` are dimensions of the image data. ``np`` is the number of
    "planes" computed as ``(number of input images - 1) // 32 + 1``. If a bit at
    position ``k`` in a pixel with coordinates ``(p, y, x)`` is 0, then input
    image number ``32 * p + k`` (0-indexed) did not contribute to the output
    data pixel with array coordinates ``(y, x)`` and if that bit is 1, then
    input image number ``32 * p + k`` did contribute to the pixel ``(y, x)``
    in the resampled image.

    As an example, let's assume we have 8 input images. Then, when ``out_ctx``
    pixel values are displayed using binary representation (and decimal in
    parenthesis), one could see values like this::

        00000001 (1) - only first input image contributed to this output pixel;
        00000010 (2) - 2nd input image contributed;
        00000100 (4) - 3rd input image contributed;
        10000000 (128) - 8th input image contributed;
        10000100 (132=128+4) - 3rd and 8th input images contributed;
        11001101 (205=1+4+8+64+128) - input images 1, 3, 4, 7, 8 have contributed
        to this output pixel.

    In order to test if a specific input image contributed to an output pixel,
    one needs to use bitwise operations. Using the example above, to test
    whether input images number 4 and 5 have contributed to the output pixel
    whose corresponding ``out_ctx`` value is 205 (11001101 in binary form) we
    can do the following:

    >>> bool(205 & (1 << (5 - 1)))  # (205 & 16) = 0 (== 0 => False): did NOT contribute
    False
    >>> bool(205 & (1 << (4 - 1)))  # (205 & 8) = 8 (!= 0 => True): did contribute
    True

    In general, to get a list of all input images that have contributed to an
    output resampled pixel with image coordinates ``(x, y)``, and given a
    context array ``ctx``, one can do something like this:

    .. doctest-skip::

        >>> import numpy as np
        >>> np.flatnonzero([v & (1 << k) for v in ctx[:, y, x] for k in range(32)])

    For convenience, this functionality was implemented in the
    :py:func:`~drizzle.utils.decode_context` function.

    Output DQ Image
    ---------------

    If DQ array of input image pixels is provided via ``dq`` parameter of
    :py:meth:`add_image`, then an output DQ array will be computed by combining
    (using bitwise-OR) DQ bitfields of all input pixels that contribute to
    a given output pixel.

    References
    ----------
    A full description of the drizzling algorithm can be found in
    `Fruchter and Hook, PASP 2002 <https://doi.org/10.1086/338393>`_.

    Examples
    --------
    .. highlight:: python
    .. code-block:: python

        # wcs1 - WCS of the input image usually with distortions (to be resampled)
        # wcs2 - WCS of the output image without distortions

        import numpy as np
        from drizzle.resample import Drizzle
        from drizzle.utils import calc_pixmap

        # simulate some data and a pixel map:
        data = np.ones((240, 570))
        pixmap = calc_pixmap(wcs1, wcs2)
        # or simulate a mapping from input image to output image frame:
        # y, x = np.indices((240, 570), dtype=np.float64)
        # pixmap = np.dstack([x, y])

        # initialize Drizzle object
        d = Drizzle(out_shape=(240, 570))
        d.add_image(data, exptime=15, pixmap=pixmap)

        # access outputs:
        d.out_img
        d.out_ctx
        d.out_wht

    """

    def __init__(
        self,
        kernel="square",
        fillval=None,
        fillval2=None,
        out_shape=None,
        out_img=None,
        out_wht=None,
        out_ctx=None,
        out_img2=None,
        out_dq=None,
        exptime=0.0,
        begin_ctx_id=0,
        max_ctx_id=None,
        disable_ctx=False,
    ):
        """
        kernel: str, optional
            The name of the kernel used to combine the input. The choice of
            kernel controls the distribution of flux over the kernel. The kernel
            names are: "square", "gaussian", "point", "turbo",
            "lanczos2", and "lanczos3". The square kernel is the default.

            .. warning::
               The "gaussian" and "lanczos2/3" kernels **DO NOT**
               conserve flux.

        out_shape : tuple, None, optional
            Shape (`numpy` order ``(Ny, Nx)``) of the output images (context
            image will have a third dimension of size proportional to the number
            of input images). This parameter is helpful when neither
            ``out_img``, ``out_wht``, nor ``out_ctx`` images are provided.

        fillval: float, None, str, optional
            The value of output pixels that did not have contributions from
            input images' pixels. When ``fillval`` is either `None` or
            ``"INDEF"`` and ``out_img`` is provided, the values of ``out_img``
            will not be modified. When ``fillval`` is either `None` or
            ``"INDEF"`` and ``out_img`` is **not provided**, the values of
            ``out_img`` will be initialized to `numpy.nan`. If ``fillval``
            is a string that can be converted to a number, then the output
            pixels with no contributions from input images will be set to this
            ``fillval`` value.

        fillval2: float, None, str, optional
            Same as ``fillval`` but applies to ``out_img2``.

        out_img : 2D array of float32, None, optional
            A 2D numpy array containing the output image produced by
            drizzling. On the first call the array values should be set to zero.
            On subsequent calls it will hold the intermediate results.

        out_img2 : 2D array of float32, list of 2D arrays of float32, None, optional
            A 2D numpy array containing the output image produced by
            drizzling *with squared weights*. This is useful when performing
            standard error propagation using variance arrays. On the first call
            the array values should be set to zero. On subsequent calls it will
            hold the intermediate results.

            Multiple output arrays (of the same shape as that of ``out_img``)
            can be provided as a list of 2D arrays. The number of arrays must
            match the number of data arrays that will be resampled and co-added
            using squared weights (see argument ``data2`` in `add_data`.)

            If ``out_img2`` is None, output arrays for the squared weights
            co-adds will be created after the first call to `add_image` based
            on the number of ``data2`` arrays.

        out_wht : 2D array of float32, None, optional
            A 2D numpy array containing the output counts. On the first
            call it should be set to zero. On subsequent calls it will
            hold the intermediate results.

        out_ctx : 2D or 3D array of int32, None, optional
            A 2D or 3D numpy array holding a bitmap of which image was an input
            for each output pixel. Should be integer zero on first call.
            Subsequent calls hold intermediate results. This parameter is
            ignored when ``disable_ctx`` is `True`.

        out_dq : 2D array of uint32, None, optional
            A 2D `~numpy.ndarray` containing DQ bitfields of output (resampled)
            pixels. It will be computed by combining (using bitwise-OR)
            DQ bitfields of input pixels that contributed to the output pixel.
            If provided, it must be a 2D array of the same shape as
            ``out_img`` and `numpy.uint32` type (unsigned 32-bit integer type).
            If `None`, output DQ array will be created during
            the first call to `add_image` and will be initialized to zero.

            .. warning::
                64-bit integer type is not supported and will raise
                an exception. Contact the authors to add support for 64-bit DQ
                if you need it.

        exptime : float, optional
            Exposure time of previously resampled images when provided via
            parameters ``out_img`` and ``out_wht``.

        begin_ctx_id : int, optional
            The context ID number (0-based) of the first image that will be
            resampled (using `add_image`). Subsequent images will be assigned
            consecutively increasing ID numbers. This parameter is ignored
            when ``disable_ctx`` is `True`.

        max_ctx_id : int, None, optional
            The largest integer context ID that is *expected* to be used for
            an input image. When it is a non-negative number and ``out_ctx`` is
            `None`, it allows to pre-allocate the necessary array for the output
            context image. If the actual number of input images that will be
            resampled will exceed initial allocation for the context image,
            additional context planes will be added as needed (context array
            will "grow" in the third dimension as new input images are added.)
            The default value of `None` is equivalent to setting ``max_ctx_id``
            equal to ``begin_ctx_id``. This parameter is ignored either when
            ``out_ctx`` is provided or when ``disable_ctx`` is `True`.

        disable_ctx : bool, optional
            Indicates to not create a context image. If ``disable_ctx`` is set
            to `True`, parameters ``out_ctx``, ``begin_ctx_id``, and
            ``max_ctx_id`` will be ignored.

        """
        self._ncoadds = 0
        self._out_img2 = None
        self._out_dq = None
        self._disable_ctx = disable_ctx

        if disable_ctx:
            self._ctx_id = None
            self._max_ctx_id = None
        else:
            if begin_ctx_id < 0:
                raise ValueError("Invalid context image ID")
            self._ctx_id = begin_ctx_id  # the ID of the *last* image to be resampled
            if max_ctx_id is None:
                max_ctx_id = begin_ctx_id
            elif max_ctx_id < begin_ctx_id:
                raise ValueError("'max_ctx_id' cannot be smaller than 'begin_ctx_id'.")
            self._max_ctx_id = max_ctx_id

        if exptime < 0.0:
            raise ValueError("Exposure time must be non-negative.")

        if exptime > 0.0 and out_img is None and out_ctx is None and out_wht is None:
            raise ValueError(
                "Exposure time must be 0.0 for the first resampling "
                "(when no output resampled images have been provided)."
            )

        if exptime == 0.0 and (
            (out_ctx is not None and np.sum(out_ctx) > 0)
            or (out_wht is not None and np.sum(out_wht) > 0)
        ):
            raise ValueError(
                "Inconsistent exposure time and context and/or weight images: "
                "Exposure time cannot be 0 when context and/or weight arrays "
                "are non-zero."
            )

        self._texptime = exptime

        if kernel.lower() not in SUPPORTED_DRIZZLE_KERNELS:
            raise ValueError(f"Kernel '{kernel}' is not supported.")
        self._kernel = kernel

        self._fillval = _process_fillval(out_img, fillval)
        self._fillval2 = _process_fillval(out_img2, fillval2)

        # shapes will collect user specified 'out_shape' and shapes of
        # out_* arrays (if provided) in order to check all shapes are the same.
        shapes = set()

        if out_img is not None:
            out_img = np.asarray(out_img, dtype=np.float32)
            shapes.add(out_img.shape)

        if out_wht is not None:
            out_wht = np.asarray(out_wht, dtype=np.float32)
            shapes.add(out_wht.shape)

        if out_ctx is not None:
            out_ctx = np.asarray(out_ctx, dtype=np.int32)
            if out_ctx.ndim == 2:
                out_ctx = out_ctx[None, :, :]
            elif out_ctx.ndim != 3:
                raise ValueError("'out_ctx' must be either a 2D or 3D array.")
            shapes.add(out_ctx.shape[1:])

        if out_dq is not None:
            t = np.min_scalar_type(out_dq)
            if t.kind not in ["i", "u"] or t.itemsize > 4:
                raise TypeError(
                    "'out_dq' must be of an unsigned integer type with itemsize of 4 bytes or less."
                )
            out_dq = np.asarray(out_dq, dtype=np.uint32)
            shapes.add(out_dq.shape)
            self._out_dq = out_dq

        if out_shape is not None:
            shapes.add(tuple(out_shape))

        if len(shapes) == 1:
            self._out_shape = shapes.pop()
            self._alloc_output_arrays(
                out_shape=self._out_shape,
                max_ctx_id=max_ctx_id,
                out_img=out_img,
                out_wht=out_wht,
                out_ctx=out_ctx,
            )

        elif len(shapes) > 1:
            raise ValueError(
                "Inconsistent data shapes specified: 'out_shape' and/or "
                "out_img, out_img2, out_wht, out_ctx, out_dq have different "
                "shapes."
            )
        else:
            self._out_shape = None
            self._out_img = None
            self._out_wht = None
            self._out_ctx = None

        if out_img2 is not None:
            if self._out_shape is not None:
                shapes.add(self._out_shape)
            if isinstance(out_img2, np.ndarray):
                out_img2 = np.asarray(out_img2, dtype=np.float32)
                shapes.add(out_img2.shape)
            else:
                for img in out_img2:
                    if img is not None:
                        shapes.add(np.shape(img))
            if len(shapes) > 1:
                raise ValueError(
                    "Inconsistent data shapes specified: 'out_shape' "
                    "and/or out_img, out_img2, out_wht, out_ctx have "
                    "different shapes."
                )
        self._output_shapes = shapes
        self._alloc_output_arrays2_init(out_img2=out_img2)

    @property
    def fillval(self):
        """Fill value for output pixels without contributions from input images."""
        return self._fillval

    @property
    def fillval2(self):
        """Fill value for output pixels in ``out_img2`` without contributions
        from input images.

        """
        return self._fillval2

    @property
    def kernel(self):
        """Resampling kernel."""
        return self._kernel

    @property
    def ctx_id(self):
        """Context image "ID" (0-based ) of the next image to be resampled."""
        return self._ctx_id

    @property
    def out_img(self):
        """Output resampled image."""
        return self._out_img

    @property
    def out_wht(self):
        """Output weight image."""
        return self._out_wht

    @property
    def out_ctx(self):
        """Output "context" image."""
        return self._out_ctx

    @property
    def out_img2(self):
        """Output resampled image(s) obtained with squared weights.
        It is always a list of one or more 2D arrays.

        """
        return self._out_img2

    @property
    def out_dq(self):
        """Output DQ image computed by OR-combining DQ bitfields of input
        images' pixels that have contributed to a given output pixel.

        """
        return self._out_dq

    @property
    def total_exptime(self):
        """Total exposure time of all resampled images."""
        return self._texptime

    def _alloc_output_arrays(self, out_shape, max_ctx_id, out_img, out_wht, out_ctx):
        # allocate arrays as needed:
        if out_wht is None:
            self._out_wht = np.zeros(out_shape, dtype=np.float32)
        else:
            self._out_wht = out_wht

        if self._disable_ctx:
            self._out_ctx = None
        else:
            if out_ctx is None:
                n_ctx_planes = max_ctx_id // CTX_PLANE_BITS + 1
                ctx_shape = (n_ctx_planes,) + out_shape
                self._out_ctx = np.zeros(ctx_shape, dtype=np.int32)
            else:
                self._out_ctx = out_ctx

            if not (out_wht is None and out_ctx is None):
                # check that input data make sense: weight of pixels with
                # non-zero context values must be different from zero:
                if np.any(np.bitwise_xor(self._out_wht > 0.0, np.sum(self._out_ctx, axis=0) > 0)):
                    raise ValueError(
                        "Inconsistent values of supplied 'out_wht' and "
                        "'out_ctx' arrays. Pixels with non-zero context "
                        "values must have positive weights and vice-versa."
                    )

        if out_img is None:
            if self._fillval.upper() in ["INDEF", "NAN"]:
                fillval = np.nan
            else:
                fillval = float(self._fillval)
            self._out_img = np.full(out_shape, fillval, dtype=np.float32)
        else:
            self._out_img = out_img

    def _alloc_output_arrays2_init(self, out_img2=None):
        if hasattr(self, "_out_img2") and self._out_img2 is not None:
            raise AssertionError(
                "It is expected that _alloc_output_arrays2_init is called "
                "before Drizzle._out_img2 is set."
            )

        if out_img2 is None:
            return

        if isinstance(out_img2, np.ndarray):
            out_img2 = [out_img2]

        self._out_img2 = []

        if isinstance(self._fillval2, str) and self._fillval2.strip().upper() == "INDEF":
            fv = np.nan
        else:
            fv = np.float32(self._fillval2)

        for i2 in out_img2:
            if i2 is None:
                if self._out_shape is None:
                    if len(self._output_shapes) == 1:
                        shape = next(iter(self._output_shapes))
                    else:
                        self._out_img2.append(None)
                        continue
                else:
                    shape = self._out_shape
                arr = np.full(shape, fill_value=fv, dtype=np.float32)
            else:
                arr = np.asarray(i2, dtype=np.float32)
            self._out_img2.append(arr)
            del arr

    def _alloc_output_arrays2_add(self, ninputs2=None):
        if isinstance(self._fillval2, str) and self._fillval2.strip().upper() == "INDEF":
            fv = np.nan
        else:
            fv = np.float32(self._fillval2)
        if self._out_img2 is None:
            if ninputs2 is None or ninputs2 < 1:
                # nothing to do
                return
            if self._ncoadds > 0:
                raise ValueError(
                    "Mismatch between the number of 'out_img2' images and the number of inputs."
                )
            self._out_img2 = [
                np.full(self._out_shape, fill_value=fv, dtype=np.float32) for _ in range(ninputs2)
            ]

        else:
            nimg2 = len(self._out_img2)

            # replace None values with arrays of _out_shape:
            for k, img in enumerate(self._out_img2):
                if img is None:
                    self._out_img2[k] = np.full(self._out_shape, fill_value=fv, dtype=np.float32)

            if (ninputs2 is not None and ninputs2 != nimg2) or (ninputs2 is None and nimg2 > 0):
                raise ValueError(
                    "Mismatch between the number of 'out_img2' images "
                    "previously set and the number of inputs."
                )

    def _increment_ctx_id(self):
        """
        Returns a pair of the *current* plane number and bit number in that
        plane and increments context image ID
        (after computing the return value).
        """
        if self._disable_ctx:
            return None, 0

        self._plane_no = self._ctx_id // CTX_PLANE_BITS
        depth = self._out_ctx.shape[0]

        if self._plane_no >= depth:
            # Add a new plane to the context image if planeid overflows
            plane = np.zeros((1,) + self._out_shape, np.int32)
            self._out_ctx = np.append(self._out_ctx, plane, axis=0)

        plane_info = (self._plane_no, self._ctx_id % CTX_PLANE_BITS)
        # increment ID for the *next* image to be added:
        self._ctx_id += 1

        return plane_info

    def add_image(
        self,
        data,
        exptime,
        pixmap,
        data2=None,
        dq=None,
        scale=_DEPRECATED_ARG,
        iscale=1.0,
        pixel_scale_ratio=1.0,
        weight_map=None,
        wht_scale=1.0,
        pixfrac=1.0,
        in_units="cps",
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
    ):
        """
        Resample and add an image to the cumulative output image. Also, update
        output total weight image and context images.

        Parameters
        ----------
        data : 2D numpy.ndarray
            A 2D numpy array containing the input image to be drizzled.

        exptime : float
            The exposure time of the input image, a positive number. The
            exposure time is used to scale the image if the units are counts.

        pixmap : 3D array
            A mapping from input image (``data``) coordinates to resampled
            (``out_img``) coordinates. ``pixmap`` must be an array of shape
            ``(Ny, Nx, 2)`` where ``(Ny, Nx)`` is the shape of the input image.
            ``pixmap[..., 0]`` forms a 2D array of X-coordinates of input
            pixels in the output frame and ``pixmap[..., 1]`` forms a 2D array of
            Y-coordinates of input pixels in the output coordinate frame.

        data2 : 2D array of float32, list of 2D arrays of float32 or None, None, optional
            A 2D numpy array (or a list of 2D arrays) with image data to be
            resampled and co-added using squared weights. The resampled output
            image can be accessed via ``out_img2`` property of the `Drizzle`
            object. This is useful for performing standard error propagation
            using variance arrays.

            Multiple data arrays (of the same shape as that of ``data``)
            can be provided as a list of 2D arrays. The number of arrays must
            match the number of output data arrays provided during
            initialization via argument ``out_img2``. If an item in the list
            is `None`, that item will not be resampled to the corresponding
            ``out_img2`` element.

            .. note::
                It is assumed that data in ``data2`` have squared units of
                ``data``. Therefore, when ``in_units`` are "counts",
                ``data2`` arrays will be rescaled by ``exptime**2`` to convert
                to rate units before resampling.

        dq : 2D array, None, optional
            A 2D numpy array of type `numpy.uint32` (unsigned 32-bit integer
            type) containing DQ bitfields of input pixels. It must
            have the same shape as ``data``. If provided, output DQ array
            (accessible via ``out_dq`` property) will be computed by combining
            (using bitwise-OR) DQ bitfields of input pixels that contributed to
            the output pixel. If `None`, DQ array of the output image will
            not be computed.

            .. warning::
                64-bit integer type is not supported and will raise
                an exception. Contact the authors to add support for 64-bit DQ
                if you need it.

        scale : float, optional
            Deprecated: use ``iscale`` and ``pixel_scale_ratio`` instead.
            It is a factor used both to rescale input image data
            by ``scale**2`` AND to compute the correct kernel size for some
            kernels ("turbo", "gaussian", and "lanczos"). It is recommended
            ``scale`` be set to pixel scale ratio: the linear dimension of
            a side of an output pixel relative to the size of an input pixel
            (or size of an output pixel in the input image's coordinate system).

        iscale : float, optional
            It is a multiplicative factor used to rescale input image data
            by ``iscale`` value. ``data2`` images will be rescaled by
            ``iscale**2``. It may make sense to rescale input image (``data``)
            by squared pixel scale ratio (the linear dimension of a side of an
            output pixel as seen in the input image's coordinate frame)
            depending on the units of the input image, i.e., counts vs
            brightness. For more details see section
            "Scaling of input image data" in :py:class:`Drizzle`.

        pixel_scale_ratio : float, None, optional
            It is a factor used to compute the correct kernel size in output
            image's coordinate system for some of the kernels
            ("turbo", "gaussian", and "lanczos") from their nominal
            sizes in input image pixels. For example, for the "lanczos3"
            kernel, the nominal size is 3 input pixels. It is recommended that
            ``pixel_scale_ratio`` be set to pixel scale ratio: the linear dimension of
            output pixel relative to the size of an input pixel. When
            ``pixel_scale_ratio`` is `None`, it will be estimated from ``pixmap`` but this
            can impose a performance penalty.

        weight_map : 2D array, None, optional
            A 2D numpy array containing the pixel by pixel weighting.
            Must have the same dimensions as ``data``.

            When ``weight_map`` is `None`, the weight of input data pixels will
            be assumed to be 1.

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
            maximum will be set in the y dimension, the full x dimension
            of the output image is the bounding box.

        Returns
        -------
        nskip : float
            The number of lines from the box defined by
            ``((xmin, xmax), (ymin, ymax))`` in the input image that were
            ignored and did not contribute to the output image.

        nmiss : float
            The number of pixels from the box defined by
            ``((xmin, xmax), (ymin, ymax))`` in the input image that were
            ignored and did not contribute to the output image.

        """
        if scale is not _DEPRECATED_ARG:
            warnings.warn(
                "Argument 'scale' has been deprecated since version 3.0 and "
                "it will be removed in a future release. "
                "Use 'iscale' and 'pixel_scale_ratio' instead and set iscale=pixel_scale_ratio**2 "
                "to achieve the same effect as with 'scale'.",
                DeprecationWarning,
            )
            iscale = scale * scale
            pixel_scale_ratio = scale

        # this enables initializer to not need output image shape at all and
        # set output image shape based on output coordinates from the pixmap.
        #
        if self._out_shape is None:
            nshapes = len(self._output_shapes)
            if nshapes == 0:
                pmap_xmin = int(np.floor(np.nanmin(pixmap[:, :, 0])))
                pmap_xmax = int(np.ceil(np.nanmax(pixmap[:, :, 0])))
                pmap_ymin = int(np.floor(np.nanmin(pixmap[:, :, 1])))
                pmap_ymax = int(np.ceil(np.nanmax(pixmap[:, :, 1])))
                pixmap = pixmap.copy()
                pixmap[:, :, 0] -= pmap_xmin
                pixmap[:, :, 1] -= pmap_ymin
                self._out_shape = (pmap_xmax - pmap_xmin + 1, pmap_ymax - pmap_ymin + 1)
            elif nshapes == 1:
                self._out_shape = next(iter(self._output_shapes))
            else:
                raise ValueError(
                    "Inconsistent data shapes: 'out_shape' and/or "
                    "out_img, out_img2, out_wht, out_ctx have different shapes."
                )  # pragma: no cover

            self._alloc_output_arrays(
                out_shape=self._out_shape,
                max_ctx_id=self._max_ctx_id,
                out_img=None,
                out_wht=None,
                out_ctx=None,
            )

        if data2 is None:
            ninputs2 = None
        else:
            if isinstance(data2, np.ndarray):
                ninputs2 = 1
                if data2.shape != data.shape:
                    raise ValueError("'data2' shape is not consistent with 'data' shape.")
            else:
                shapes2 = set()
                ninputs2 = len(data2)
                data2 = list(data2)
                for k, d in enumerate(data2):
                    if d is None or d.size == 0:
                        data2[k] = None
                    else:
                        shapes2.add(d.shape)

                if (len(shapes2) == 1 and shapes2.pop() != data.shape) or len(shapes2) > 1:
                    raise ValueError("'data2' shape(s) is not consistent with 'data' shape.")

        self._alloc_output_arrays2_add(ninputs2=ninputs2)

        plane_no, id_in_plane = self._increment_ctx_id()

        if exptime <= 0.0:
            raise ValueError("'exptime' *must* be a strictly positive number.")

        # Ensure that the fillval parameter gets properly interpreted
        # for use with tdriz
        if in_units == "cps":
            expscale = 1.0
        else:
            expscale = exptime

        self._texptime += exptime

        data = np.asarray(data, dtype=np.float32)
        pixmap = np.asarray(pixmap, dtype=np.float64)
        in_ymax, in_xmax = data.shape

        if pixmap.shape[:2] != data.shape:
            raise ValueError("'pixmap' shape is not consistent with 'data' shape.")

        if xmin is None or xmin < 0:
            xmin = 0

        if ymin is None or ymin < 0:
            ymin = 0

        if xmax is None or xmax > in_xmax - 1:
            xmax = in_xmax - 1

        if ymax is None or ymax > in_ymax - 1:
            ymax = in_ymax - 1

        if weight_map is not None:
            weight_map = np.asarray(weight_map, dtype=np.float32)
            if weight_map.shape != data.shape:
                raise ValueError("'weight_map' shape is not consistent with 'data' shape.")
        else:  # TODO: this should not be needed after C code modifications
            weight_map = np.ones_like(data)

        pixmap = np.asarray(pixmap, dtype=np.float64)

        if self._disable_ctx:
            ctx_plane = None
        else:
            if self._out_ctx.ndim == 2:
                raise AssertionError("Context image is expected to be 3D")
            ctx_plane = self._out_ctx[plane_no]

        if dq is not None:
            t = np.min_scalar_type(dq)
            if t.kind not in ["i", "u"] or t.itemsize > 4:
                raise TypeError(
                    "'dq' must be of an unsigned integer type with itemsize of 4 bytes or less."
                )
            dq = np.asarray(dq, dtype=np.uint32)
            if dq.shape != data.shape:
                raise ValueError("'dq' shape is not consistent with 'data' shape.")
            if self._out_dq is None:
                self._out_dq = np.zeros(self._out_shape, dtype=np.uint32)

        # TODO: probably tdriz should be modified to not return version.
        #       we should not have git, Python, C, ... versions

        # TODO: While drizzle code in cdrizzlebox.c supports weight_map=None,
        #       cdrizzleapi.c does not. It should be modified to support this
        #       for performance reasons.

        _vers, nmiss, nskip = cdrizzle.tdriz(
            input=data,
            weights=weight_map,
            pixmap=pixmap,
            output=self._out_img,
            counts=self._out_wht,
            context=ctx_plane,
            input2=data2,
            output2=self._out_img2,
            dq=dq,
            outdq=self._out_dq,
            uniqid=id_in_plane + 1,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            iscale=iscale,  # scales image intensity. usually equal to 1 or
            # (pixel scale ratio)**2
            pscale_ratio=pixel_scale_ratio,  # scales kernel size. usually equal to pixel scale ratio
            pixfrac=pixfrac,
            kernel=self._kernel,
            in_units=in_units,
            expscale=expscale,
            wtscale=wht_scale,
            fillstr=self._fillval,
            fillstr2=self._fillval2,
        )
        self._cversion = _vers  # TODO: probably not needed
        self._ncoadds += 1

        return nmiss, nskip


def blot_image(
    data,
    pixmap,
    pix_ratio=_DEPRECATED_ARG,
    exptime=_DEPRECATED_ARG,
    output_pixel_shape=_DEPRECATED_ARG,
    out_img=None,
    fillval=0.0,
    iscale=1.0,
    interp="poly5",
    sinscl=1.0,
):
    """
    Resample the ``data`` input image onto an output grid defined by
    the ``pixmap`` array. ``blot_image`` performs resampling using one of
    the several interpolation algorithms and, unlike the "drizzle" algorithm
    with 'square', 'turbo', and 'point' kernels, this resampling is not
    flux-conserving.

    This method works best for with well sampled images and thus it is
    typically used to resample the output of :py:class:`Drizzle` back to the
    coordinate grids of input images of :py:meth:`Drizzle.add_image`.
    The output of :py:class:`Drizzle` are usually well sampled images especially
    if it was created from a set of dithered images.

    Parameters
    ----------
    data : 2D array
        Input numpy array of the source image in units of 'cps'.

    pixmap : 3D array
        A mapping from input image (``data``) coordinates to resampled
        (``out_img``) coordinates. ``pixmap`` must be an array of shape
        ``(Ny, Nx, 2)`` where ``(Ny, Nx)`` is the shape of the input image.
        ``pixmap[..., 0]`` forms a 2D array of X-coordinates of input
        pixels in the output frame and ``pixmap[..., 1]`` forms a 2D array of
        Y-coordinates of input pixels in the output coordinate frame.

    pix_ratio : float
        Ratio of the input image pixel scale to the output image pixel scale as
        used in the ``drizzle`` context: input is a distorted image that was
        "drizzled" onto the output image. That is, it is the ratio of the
        scale of the pixels in the input ``data`` argument to the scale of
        pixels of the image array returned by ``blot_image()``.
        **It is used to scale the input image intensities to account
        for the change in pixel area.**

        .. warning::
            Deprecated since version 3.0 and will be removed in a future
            release. Use ``iscale`` instead and set
            ``iscale=1.0 / pix_ratio**2`` to achieve the same effect as with
            ``pix_ratio``.

    exptime : float
        The exposure time of the input image. If provided it is used to scale
        the output image values.

        .. warning::
            Deprecated since version 3.0 and will be removed in a future
            release. Use ``iscale`` instead and set
            ``iscale=exptime`` or ``exptime / pix_ratio**2`` to achieve the
            same effect as with ``exptime`` (and ``pix_ratio``).

    output_pixel_shape : tuple of int
        A tuple of two integer numbers indicating the dimensions of the output
        image ``(Nx, Ny)``.

        .. warning::
            Deprecated since version 3.0 and will be removed in a future
            release. It is not needed since the output image shape can be
            inferred from ``pixmap``.

    output_image : 2D array of float32, None, optional
        A 2D numpy array to hold the output image produced by resampling
        the input image (``data``). If `None`, a new array will be allocated.

    fillval: float, optional
        The value of output pixels that did not have contributions from
        input image' pixels.

    iscale : float, optional
        A multiplicative factor used to rescale output image data by
        ``iscale``. Depending on specific needs, it may make sense to rescale
        output image by inverse of squared pixel scale ratio (the linear
        dimension of a side of a resampled/drizzled (input) pixel as seen in
        the distorted (output) image's coordinate frame) depending on the units
        of the input image, i.e., counts (flux) vs surface brightness.
        For more details see section "Scaling of input image data" in
        :py:class:`Drizzle`.

    interp : str, optional
        The type of interpolation used in the resampling. The
        possible values are:

            - "nearest" (nearest neighbor interpolation);
            - "linear" (bilinear interpolation);
            - "poly3" (cubic polynomial interpolation);
            - "poly5" (quintic polynomial interpolation);
            - "sinc" (sinc interpolation);
            - "lan3" (3rd order Lanczos interpolation); and
            - "lan5" (5th order Lanczos interpolation).

        .. warning::
            The "sinc" interpolation is currently investigated for possible
            issues, see https://github.com/spacetelescope/drizzle/issues/209,
            and its use is not recommended. Furthermore, sinc interpolation may
            be removed in future releases.

    sincscl : float, optional
        The scaling factor for "sinc" interpolation.

    Returns
    -------
    out_img : 2D numpy.ndarray
        A 2D numpy array containing the resampled image data.

    """
    if pix_ratio is not _DEPRECATED_ARG:
        warnings.warn(
            "Argument 'pix_ratio' has been deprecated since version 3.0 and "
            "it will be removed in a future release. "
            "Use 'iscale' instead and set iscale=1.0 / pix_ratio**2 "
            "to achieve the same effect as with 'pix_ratio'.",
            DeprecationWarning,
        )
        iscale /= pix_ratio * pix_ratio

    if exptime is not _DEPRECATED_ARG:
        warnings.warn(
            "Argument 'exptime' has been deprecated since version 3.0 and "
            "it will be removed in a future release. "
            "Use 'iscale' instead and set iscale=exptime "
            "to achieve the same effect as with 'exptime'.",
            DeprecationWarning,
        )
        iscale *= exptime

    if output_pixel_shape is _DEPRECATED_ARG:
        output_shape = tuple(pixmap.shape[:2])
    else:
        warnings.warn(
            "Argument 'output_pixel_shape' has been deprecated since version "
            "3.0 and it will be removed in a future release. It is not needed "
            "since the output image shape can be inferred from 'pixmap'.",
            DeprecationWarning,
        )
        output_shape = output_pixel_shape[::-1]

    if out_img is None:
        out_img = np.empty(output_shape, dtype=np.float32)
    else:
        out_img = np.asarray(out_img, dtype=np.float32)
        if out_img.shape != output_shape:
            raise ValueError("'output_image' shape is not consistent with 'pixmap' shape.")

    cdrizzle.tblot(
        data, pixmap, out_img, iscale=iscale, interp=interp, fillval=fillval, sinscl=sinscl
    )

    return out_img


def _process_fillval(out_img, fillval):
    if fillval is None:
        fillval = "INDEF"

    elif isinstance(fillval, str):
        fillval = fillval.strip()
        if fillval.upper() in ["", "INDEF"]:
            fillval = "INDEF"
        else:
            float(fillval)
            fillval = str(fillval)

    else:
        fillval = str(fillval)

    if out_img is None and fillval == "INDEF":
        fillval = "NaN"

    return fillval
