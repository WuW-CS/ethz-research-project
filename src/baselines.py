"""
baselines.py - simple reference methods the learned model has to beat.

Common signature: every method takes the low-resolution image `lr` (h, w) and the integer
`factor`, and returns a high-resolution image of shape (h * factor, w * factor).
Methods that use a guide also take `guide`, an array of the HIGH-resolution shape
(h * factor, w * factor), e.g. the DEM or GHSL raster (see data.load_guide).

"""
import numpy as np
import scipy as sp


def cubic(lr, factor):
    """Cubic interpolation, no guide.

    Docs / options to compare: scipy.ndimage.zoom(order=3), cv2.resize(INTER_CUBIC),
    skimage.transform.resize(order=3). They differ in kernel, edge handling and pre-filtering:
    pick ONE and write the choice in your report.
    """
    raise NotImplementedError


def joint_bilateral(lr, guide, factor, sigma_spatial=1.0, sigma_range=0.1, radius=2):
    """Joint bilateral upsampling (Kopf et al., 2007).

    For every HR pixel p: weighted mean of nearby LR values q, with weight
        w(p, q) = exp(-|p - q|^2 / (2 sigma_spatial^2)) * exp(-(g(p) - g(q))^2 / (2 sigma_range^2))
    where |p - q| is measured in LR pixels and g(.) is the guide (g(q) read at the HR location
    that corresponds to the LR pixel q, e.g. its block mean). Normalize by the sum of weights.

    Hint: a short NumPy implementation (loops or shifted arrays) is enough; no library needed.
    """
    raise NotImplementedError


def guided_filter(lr, guide, factor, radius=4, eps=1e-3):
    """Guided filter (He et al., 2013) used for upsampling.

    Steps: (1) upsample `lr` to the HR grid with `cubic`; (2) filter that image with `guide`
    as the guidance image. The filter assumes q = a * guide + b inside each window; a and b come
    from box-filtered means/variances/covariances.

    Options: cv2.ximgproc.guidedFilter (opencv-contrib-python) or a NumPy version
    (e.g. github.com/lisabug/guided-filter). Check one against the other on a small example.
    """
    raise NotImplementedError