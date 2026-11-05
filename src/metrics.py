"""
metrics.py - one place where "how good is this result" is defined, for every method.

All functions ignore pixels where `pred` or `ref` is NaN (use the same mask for all methods,
otherwise the comparison is unfair). baselines.py puts NaN exactly where `lr` is NaN, so every
method is evaluated on the same pixels.
"""
import numpy as np

import data


def _valid_mask(pred, ref):
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {ref.shape}")
    return np.isfinite(pred) & np.isfinite(ref)


def _masked(pred, ref):
    """Valid pixels of both arrays as two 1D vectors (None, None if there is none)."""
    m = _valid_mask(pred, ref)
    if not m.any():
        return None, None
    return np.asarray(pred, float)[m], np.asarray(ref, float)[m]


def rmse(pred, ref):
    """Root mean squared error over valid pixels, in the units of the data (mol/m2)."""
    p, r = _masked(pred, ref)
    if p is None:
        return np.nan
    return float(np.sqrt(np.mean((p - r) ** 2)))


def mae(pred, ref):
    """Mean absolute error over valid pixels."""
    p, r = _masked(pred, ref)
    if p is None:
        return np.nan
    return float(np.mean(np.abs(p - r)))


def bias(pred, ref):
    """Mean signed error pred - ref (useful to see a systematic offset)."""
    p, r = _masked(pred, ref)
    if p is None:
        return np.nan
    return float(np.mean(p - r))


def psnr(pred, ref, data_range=None):
    """Peak signal-to-noise ratio (dB) over valid pixels.

    data_range : the dynamic range of the data. If None, inferred as max(ref) - min(ref) on the
    valid support. NOT scale-invariant: values are meaningless across different units.
    For a fixed `ref` it is a decreasing function of the RMSE, so it ranks methods like RMSE.
    """
    p, r = _masked(pred, ref)
    if p is None:
        return np.nan
    if data_range is None:
        data_range = float(r.max() - r.min())
    if data_range <= 0:
        return np.nan
    mse = np.mean((p - r) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def consistency_error(pred, lr, factor):
    """RMSE between block_average(pred, factor) and the observed `lr`.

    The project's hard constraint: the fine field, averaged over each coarse cell, must
    reproduce the observation. Report it for EVERY method (cubic does not satisfy it exactly).
    """
    return rmse(data.block_average(pred, factor, nan_threshold=0.0), lr)


def gradient_correlation(pred, guide):
    """Correlation between the spatial gradient magnitudes of `pred` and of `guide`.

    Diagnostic for research question 3 (texture copying): a value close to 1 with a guide that
    has no physical link to NO2 suggests the method copies guide texture into the NO2 field.
    Returns NaN if one of the two has no variation.
    """
    gp = np.hypot(*np.gradient(np.asarray(pred, dtype=np.float64)))
    gg = np.hypot(*np.gradient(np.asarray(guide, dtype=np.float64)))
    p, g = _masked(gp, gg)
    if p is None or len(p) < 2 or p.std() == 0 or g.std() == 0:
        return np.nan
    return float(np.corrcoef(p, g)[0, 1])