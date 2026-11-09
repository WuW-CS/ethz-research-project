"""
tune_guides.py - diagnostic: why do the guided filters not beat plain cubic interpolation?

1. Smoothness of the NO2 map (high-frequency share, neighbour correlation)
2. Correlation guide vs NO2: raw values (large scale) and high-pass (fine scale)
3. Oracle test: guide = hr itself (must beat cubic, otherwise there is a bug in the code)
4. Guides pre-processed (log1p / percentile clip) to avoid heavy tails
5. Parameter grid for joint bilateral (sigma_spatial, sigma_range) and guided filter (radius, eps)
6. Reference gradient correlation between hr and each guide

Run from scripts/:  python tune_guides.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import baselines, data, metrics

DATAS = Path("/scratch4/willwu/datas/madrid")
GUIDES_DIR = DATAS / "guides"
FACTOR = 4
HP_SIGMA = 3          # high-pass filter: x - gaussian(x, sigma) removes structures larger than ~3 px

hr_full = data.load_tif(DATAS / "L3_NO2_mean" / "madrid_no2_mean.tif")


def prep(arr, log=False):
    """Optional log1p, then clip to the 1st-99th percentiles (removes heavy tails)."""
    a = np.log1p(arr) if log else arr
    lo, hi = np.nanpercentile(a, [1, 99])
    return np.clip(a, lo, hi)


def high_pass(x):
    return x - ndimage.gaussian_filter(x, HP_SIGMA)


guides = {
    "dem": prep(data.load_guide(sorted(GUIDES_DIR.glob("*dem*.tif"))[0], normalize=False)),
    "ghsl_log": prep(data.load_guide(sorted(GUIDES_DIR.glob("*ghsl*.tif"))[0], normalize=False), log=True),
    "osm_dens": prep(data.load_guide(sorted(GUIDES_DIR.glob("*osm*density*.tif"))[0], normalize=False)),
    "osm_dist": prep(data.load_guide(sorted(GUIDES_DIR.glob("*osm*distance*.tif"))[0], normalize=False)),
}

# ---------------------------------------------------------------- 1. smoothness of the NO2 map
print("== NO2 map smoothness ==")
print(f"high-frequency share of the std : {np.nanstd(high_pass(hr_full)) / np.nanstd(hr_full):.3f}")
nb = np.corrcoef(hr_full[:, :-1].ravel(), hr_full[:, 1:].ravel())[0, 1]
print(f"horizontal neighbour correlation: {nb:.4f}")

# ---------------------------------------------------------------- 2. guide vs NO2 correlation
print(f"\n{'guide':<12}{'corr values':>13}{'corr high-pass':>16}")
for name, g in guides.items():
    c_val = np.corrcoef(hr_full.ravel(), g.ravel())[0, 1]
    c_hp = np.corrcoef(high_pass(hr_full).ravel(), high_pass(g).ravel())[0, 1]
    print(f"{name:<12}{c_val:>13.2f}{c_hp:>16.2f}")

# ---------------------------------------------------------------- 3-6. oracle + tuning
guides = {"oracle(hr)": hr_full, **guides}

lr, hr, _ = data.make_synthetic_case(hr_full, hr_full, FACTOR)
cubic_pred = baselines.cubic(lr, FACTOR)
print(f"\ncubic (reference) RMSE = {metrics.rmse(cubic_pred, hr):.3e}\n")

SIGMA_SPATIAL = [0.5, 1.0]
SIGMA_RANGE = [0.02, 0.05, 0.1, 0.3]
RADII = [FACTOR, 2 * FACTOR]
EPS = [1e-5, 1e-4, 1e-3, 1e-2]

print(f"{'guide':<12}{'gradcorr(hr,g)':>15}  {'best joint bilateral':<30}{'best guided filter':<30}")
for name, guide in guides.items():
    lr, hr, g = data.make_synthetic_case(hr_full, guide, FACTOR)
    # gradient correlation is meaningless for a distance map
    gc = metrics.gradient_correlation(hr, g) if name != "osm_dist" else float("nan")

    best_jb = min(
        (metrics.rmse(baselines.joint_bilateral(lr, g, FACTOR, ss, sr), hr), ss, sr)
        for ss, sr in itertools.product(SIGMA_SPATIAL, SIGMA_RANGE))
    best_gf = min(
        (metrics.rmse(baselines.guided_filter(lr, g, FACTOR, r, e), hr), r, e)
        for r, e in itertools.product(RADII, EPS))

    print(f"{name:<12}{gc:>15.2f}  "
          f"{best_jb[0]:.2e} (ss={best_jb[1]}, sr={best_jb[2]})   "
          f"{best_gf[0]:.2e} (r={best_gf[1]}, eps={best_gf[2]:g})")

print("\nHow to read it:")
print("- oracle better than cubic -> the code is correct; oracle worse than cubic -> look for a bug.")
print("- real guides still worse than cubic after tuning -> the guide does not help on this map "
      "(or the ground truth is already very smooth).")
print("- correlation on values but not on high-pass -> guides explain the large-scale level, not the fine detail.")