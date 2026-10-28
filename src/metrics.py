"""
All functions ignore pixels where `pred` or `ref` is NaN (use the same mask for all methods,
otherwise the comparison is unfair).
"""

import numpy as np

def _valid_mask(pred,ref):
    pred = np.asarray(pred,dtype=np.float64)
    ref = np.asarray(pred,dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {ref.shape}")
    return np.isfinite(pred) & np.isfinite(ref)

def _masked(pred, ref):
    m = _valid_mask(pred, ref)
    if not m.any():
        return None, None
    return np.asarray(pred, float)[m], np.asarray(ref, float)[m]


def rmse(pred, ref):
    p,r = _masked(pref,ref)
    if p is None:
        return np.nan
    return float(np.sqrt(np.mean((p-r)**2)))

def mae(pred,ref):
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