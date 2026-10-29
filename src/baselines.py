"""
baselines.py - classic reference methods for guided super-resolution, built on libraries.

Common signature: every method takes the low-resolution image `lr` (h, w) and the integer
`factor`, and returns a high-resolution image of shape (h * factor, w * factor).
Methods that use a guide also take `guide`, an array of the HIGH-resolution shape
(h * factor, w * factor), e.g. the DEM or GHSL raster.

Where each method comes from
----------------------------
- cubic            : scipy.ndimage.zoom (order=3, cubic spline).
- joint_bilateral  : OpenCV contrib, cv2.ximgproc.jointBilateralFilter, applied to the
                     nearest-neighbour upsampled input (standard way to get Kopf et al.'s
                     joint bilateral upsampling out of a joint bilateral filter).
- guided_filter    : OpenCV contrib, cv2.ximgproc.guidedFilter (He et al., 2013), applied to
                     the cubic-upsampled input. 

Conventions shared by all methods
---------------------------------
- NaN in `lr`: filled with the nearest valid value before filtering (so the filters see no hole),
  and the corresponding HR pixels are set back to NaN in the output.
- The guide is min-max normalized to [0, 1] internally, so `sigma_range` and `eps` always refer
  to a guide in [0, 1], whatever its units (m, m3...). A NaN in the guide is replaced by its mean.
- The LR values are standardized (mean 0, std 1) before filtering and un-standardized after.
  Both filters are linear in the input, so this changes nothing mathematically; it only avoids
  float32 precision problems with tiny values such as NO2 columns (~1e-5 mol/m2).
"""
import cv2
import numpy as np
from scipy import ndimage


# ----------------------------------------------------------------------------- helpers
def _fill_nan(lr):
    """Replace NaN by the nearest valid value. Returns (filled float array, valid mask)."""
    lr = np.asarray(lr, dtype=np.float64)
    valid = np.isfinite(lr)
    if not valid.any():
        raise ValueError("lr contains no valid pixel")
    if valid.all():
        return lr, valid
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return lr[idx[0], idx[1]], valid


def _restore_nan(out, valid, factor):
    """Put NaN back where the LR pixel was invalid (mask upsampled by `factor`)."""
    hr_valid = np.repeat(np.repeat(valid, factor, axis=0), factor, axis=1)
    out = out.copy()
    out[~hr_valid] = np.nan
    return out


def _prepare_guide(guide, shape):
    """Check the shape, fill NaN, min-max normalize to [0, 1]; float32 (OpenCV friendly)."""
    guide = np.asarray(guide, dtype=np.float64)
    if guide.shape != tuple(shape):
        raise ValueError(f"guide has shape {guide.shape}, expected {tuple(shape)} (= lr * factor)")
    guide = np.where(np.isfinite(guide), guide, np.nanmean(guide))
    span = guide.max() - guide.min()
    guide = (guide - guide.min()) / span if span > 0 else np.zeros_like(guide)
    return guide.astype(np.float32)


def _standardize(x):
    """Return (x - mean) / std, mean, std (std replaced by 1 for a constant image)."""
    m, s = float(np.mean(x)), float(np.std(x))
    return (x - m) / (s if s > 0 else 1.0), m, (s if s > 0 else 1.0)


def _upsample_nearest(x, factor):
    """Each LR pixel becomes a factor x factor block."""
    return np.repeat(np.repeat(x, factor, axis=0), factor, axis=1)


# ----------------------------------------------------------------------------- methods
def cubic(lr, factor):
    """Cubic interpolation, no guide (scipy.ndimage.zoom, order=3).

    grid_mode=True treats pixels as cells (each LR cell is split into factor x factor HR
    cells), which matches block-averaged data; without it, scipy aligns the corner pixel
    centres and the output is slightly shifted. mode="nearest" repeats the edge values.
    """
    filled, valid = _fill_nan(lr)
    out = ndimage.zoom(filled, factor, order=3, grid_mode=True, mode="nearest")
    return _restore_nan(out, valid, factor)


def joint_bilateral(lr, guide, factor, sigma_spatial=1.0, sigma_range=0.1):
    """Joint bilateral upsampling (Kopf et al., 2007) via cv2.ximgproc.jointBilateralFilter.

    Each HR pixel p becomes a weighted mean of the (nearest-upsampled) LR values around it:
        weight = Gauss(spatial distance, sigma_space) * Gauss(|guide(p) - guide(q)|, sigma_range)
    so neighbours that look like p in the GUIDE count more.

    sigma_spatial : spatial sigma in LR pixels (converted to HR pixels with `factor`).
    sigma_range   : sigma on the guide, which is in [0, 1] (0.05-0.2 is a reasonable range to try).
    """
    filled, valid = _fill_nan(lr)
    h, w = filled.shape
    g = _prepare_guide(guide, (h * factor, w * factor))
    z, mean, std = _standardize(filled)
    src = _upsample_nearest(z, factor).astype(np.float32)

    sigma_space = sigma_spatial * factor                  # HR pixels
    diameter = 2 * int(np.ceil(2.0 * sigma_space)) + 1    # window covers about +-2 sigma
    out = cv2.ximgproc.jointBilateralFilter(g, src, diameter, sigma_range, sigma_space)
    return _restore_nan(out.astype(np.float64) * std + mean, valid, factor)


def guided_filter(lr, guide, factor, radius=None, eps=1e-2):
    """Guided filter (He et al., 2013) used for upsampling.

    Step 1: cubic-upsample `lr` to the HR grid. Step 2: filter it with the guide. Inside each
    window the output is modelled as q = a * guide + b (a, b from local means / (co)variances).

    radius  : window radius in HR pixels (default 2 * factor, i.e. about +-2 LR pixels).
    eps     : regularization on the guide variance (guide in [0, 1]); larger = smoother,
              less guide structure transferred. 1e-3 to 1e-1 is a reasonable range to try.
    """
    filled, valid = _fill_nan(lr)
    h, w = filled.shape
    g = _prepare_guide(guide, (h * factor, w * factor))
    radius = 2 * factor if radius is None else int(radius)
    z, mean, std = _standardize(filled)
    up = ndimage.zoom(z, factor, order=3, grid_mode=True, mode="nearest").astype(np.float32)

    out = cv2.ximgproc.guidedFilter(g, up, radius, eps)
    return _restore_nan(out.astype(np.float64) * std + mean, valid, factor)





# ----------------------------------------------------------------------------- self-test LLM generated
def _selftest():
    """Synthetic check: a truth that depends partly on the guide, degraded by block averaging."""
    rng = np.random.default_rng(0)
    h_lr, w_lr, factor = 45, 58, 4
    H, W = h_lr * factor, w_lr * factor

    def smooth_noise(sigma):
        x = ndimage.gaussian_filter(rng.normal(size=(H, W)), sigma)
        return (x - x.min()) / (x.max() - x.min())

    guide = smooth_noise(3)                      # fine-scale, informative guide
    truth = 2e-5 * (1.0 + 0.8 * guide + 0.6 * smooth_noise(25))   # NO2-like magnitude
    lr = truth.reshape(h_lr, factor, w_lr, factor).mean(axis=(1, 3))

    def rmse(a):
        return float(np.sqrt(np.nanmean((a - truth) ** 2)))

    def consistency(a):
        return float(np.sqrt(np.mean((a.reshape(h_lr, factor, w_lr, factor).mean(axis=(1, 3)) - lr) ** 2)))

    results = {
        "cubic (scipy)": cubic(lr, factor),
        "joint_bilateral": joint_bilateral(lr, guide, factor),
        "guided_filter (opencv)": guided_filter(lr, guide, factor),
        
    }
    for name, out in results.items():
        assert out.shape == truth.shape, (name, out.shape)
    print(f"{'method':<26}{'RMSE vs truth':>16}{'consistency':>14}")
    for name, out in results.items():
        print(f"{name:<26}{rmse(out):>16.3e}{consistency(out):>14.3e}")
 
    # cross-check: scipy cubic vs OpenCV cubic
    cv = cv2.resize(lr.astype(np.float32), (w_lr * factor, h_lr * factor), interpolation=cv2.INTER_CUBIC)
    print(f"scipy vs cv2 cubic, RMSE of the difference: "
          f"{float(np.sqrt(np.mean((results['cubic (scipy)'] - cv) ** 2))):.2e}")

    # NaN handling: a hole in lr must give a hole (and only it) in the output
    lr_nan = lr.copy()
    lr_nan[5:8, 10:13] = np.nan
    for name, f in {"cubic": lambda: cubic(lr_nan, factor),
                    "joint_bilateral": lambda: joint_bilateral(lr_nan, guide, factor),
                    "guided_filter": lambda: guided_filter(lr_nan, guide, factor)}.items():
        out = f()
        expected = np.repeat(np.repeat(np.isnan(lr_nan), factor, 0), factor, 1)
        assert np.array_equal(np.isnan(out), expected), name
    print("NaN handling OK (holes preserved, nothing else becomes NaN)")


if __name__ == "__main__":
    _selftest()