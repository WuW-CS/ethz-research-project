"""
data.py - loading the NO2 / guide rasters and building the synthetic test (degradation).

Conventions
-----------
- Every image is a 2D float array (H, W) on the same regular grid (~1.1 km cells, EPSG:4326).
- "No data" is NaN. Exports sometimes write exact 0 for empty NO2 images: for NO2 (kind="no2")
  0 is converted to NaN. For guides (DEM, GHSL) 0 is a REAL value (0 m, nothing built): kept.
- A "factor" is the integer downsampling factor of the synthetic test (e.g. 4 -> HR cell = LR cell / 4).

Synthetic test (the only way to measure an error with the L3 data)
------------------------------------------------------------------
    hr  = the real daily NO2 map                (the "truth", shape H x W)
    lr  = block_average(hr, factor)             (what a coarse sensor would see)
    out = method(lr, guide, factor)             (shape H x W again)
    error = metric(out, hr)
"""
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (provides rioxarray.open_rasterio and the .rio accessor)


def load_tif(path, kind="no2"):
    """Read a single-band GeoTIFF -> float array (H, W), with nodata replaced by NaN.

    kind="no2": exact 0 is also converted to NaN (empty images exported as 0).
    any other kind (e.g. "guide"): 0 is a real value and is kept.
    """
    with rioxarray.open_rasterio(path) as ds:
        data = ds.values[0].astype(float)
        nodata = ds.rio.nodata
    if nodata is not None and np.isfinite(nodata):
        data[data == nodata] = np.nan
    if kind == "no2":
        data[data == 0] = np.nan
    return data


def daily_mean(paths):
    """Merge several overpasses of the same day -> one (H, W) map.

    NaN-aware mean (np.nanmean over the stacked images). Where every image is NaN, the
    result stays NaN.
    """
    stack = np.stack([load_tif(p) for p in paths], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # "Mean of empty slice"
        return np.nanmean(stack, axis=0)


def load_daily_maps(folder, min_valid=0.0):
    """Load every NO2 day of `folder` -> {YYYYMMDD: (H, W) array}.

    Files are grouped by the first 8 characters of their name (acquisition date,
    e.g. "20250905T112639_..." -> "20250905"), and each group is merged with daily_mean.
    Days whose merged map has less than `min_valid` valid pixels (fraction in [0, 1]) are
    dropped; with the default 0.0, only days with no valid pixel at all are dropped.
    """
    groups = defaultdict(list)
    for tif in sorted(Path(folder).glob("*.tif")):
        groups[tif.stem[:8]].append(str(tif))
    days = {}
    for day, paths in sorted(groups.items()):
        merged = daily_mean(paths)
        valid = np.isfinite(merged).mean()
        if valid > 0 and valid >= min_valid:
            days[day] = merged
    return days


def load_guide(path, normalize=True):
    """Read a guide raster (DEM, GHSL...) -> float array (H, W).

    If `normalize`, rescale to [0, 1] (min-max, ignoring NaN). Bilateral / guided filters
    compare guide values with each other, so their scale matters.
    """
    data = load_tif(path, kind="guide")
    if normalize:
        lo, hi = np.nanmin(data), np.nanmax(data)
        data = (data - lo) / (hi - lo) if hi > lo else np.zeros_like(data)
    return data


def block_average(img, factor, nan_threshold=0.0):
    """Downsample by averaging non-overlapping factor x factor blocks (NaN-aware).

    - H and W are cropped (top-left aligned) to a multiple of `factor` first.
    - A block becomes NaN if the fraction of NaN pixels in it is > `nan_threshold`
      (0.0 = any NaN makes the block NaN, the strictest choice). Otherwise its value is the
      mean of its valid pixels.
    Returns an array of shape (H // factor, W // factor).

    Trick: reshape (H, W) -> (h, f, w, f); axes 0 and 2 index the block, axes 1 and 3 the
    position inside it, so averaging over axes (1, 3) averages each block.
    """
    img = np.asarray(img, dtype=float)
    h, w = img.shape
    H, W = (h // factor) * factor, (w // factor) * factor
    blocks = img[:H, :W].reshape(H // factor, factor, W // factor, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN blocks
        out = np.nanmean(blocks, axis=(1, 3))
    nan_frac = np.isnan(blocks).mean(axis=(1, 3))
    out[nan_frac > nan_threshold] = np.nan
    return out


def crop_to_multiple(img, factor):
    """Crop (top-left aligned) so that H and W are multiples of `factor`.

    Use it on the truth `hr` so that out.shape == hr.shape for every method.
    """
    img = np.asarray(img)
    h, w = img.shape
    return img[:(h // factor) * factor, :(w // factor) * factor]


def crop_all(arrays, factor):
    """Crop every array to the same common shape (multiple of `factor`)."""
    h = min(a.shape[0] for a in arrays)
    w = min(a.shape[1] for a in arrays)
    H, W = (h // factor) * factor, (w // factor) * factor
    return [np.asarray(a)[:H, :W] for a in arrays]


def make_synthetic_case(hr, guide, factor, nan_threshold=0.0):
    """Return (lr, hr_cropped, guide_cropped) ready for baselines.py.

    The NO2 map and the guide must already be on the same grid (same shape): a silent crop
    would hide a 1-pixel shift between the two rasters and misalign the guide.
    """
    if hr.shape != guide.shape:
        raise ValueError(f"NO2 and guide are not on the same grid: {hr.shape} vs {guide.shape}")
    hr_c, guide_c = crop_all([hr, guide], factor)
    lr = block_average(hr_c, factor, nan_threshold=nan_threshold)
    return lr, hr_c, guide_c