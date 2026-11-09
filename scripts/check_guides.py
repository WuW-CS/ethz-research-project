"""
check_guides.py - sanity checks on the NO2 map and the guide rasters before run_baselines.py

1. Print which file is picked for each guide (DEM, GHSL, OSM density, OSM distance)
2. Print shape / transform / CRS / nodata / % NaN / min / max of every raster
3. Check that every raster is on exactly the same grid as the NO2 map
4. Save one figure: NO2 | DEM | GHSL | OSM density | OSM distance (visual check)
5. Mini baseline test (factor 4) with the four guides

Run from scripts/:  python check_guides.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless backend (SSH server)
import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import baselines, data, metrics

DATAS = Path("/scratch4/willwu/datas/madrid")
NO2_FILE = DATAS / "L3_NO2_mean" / "madrid_no2_mean.tif"
GUIDES_DIR = DATAS / "guides"


def pick(pattern):
    """First file matching the pattern; warn if several files match."""
    files = sorted(GUIDES_DIR.glob(pattern))
    if not files:
        raise SystemExit(f"No file matching {pattern} in {GUIDES_DIR}")
    extra = f"  (+{len(files) - 1} other match(es)!)" if len(files) > 1 else ""
    print(f"  {pattern:<22} -> {files[0].name}{extra}")
    return files[0]


# ---------------------------------------------------------------- 1. file selection
print("== Selected files ==")
FILES = {
    "no2": NO2_FILE,
    "dem": pick("*dem*.tif"),
    "ghsl": pick("*ghsl*.tif"),
    "osm_dens": pick("*osm*density*.tif"),
    "osm_dist": pick("*osm*distance*.tif"),
}

# ---------------------------------------------------------------- 2 + 3. grid check
print("\n== Raster info ==")
ref = None
arrays = {}
grids_ok = True
for name, path in FILES.items():
    with rasterio.open(path) as r:
        transform, crs, shape = r.transform, r.crs, r.shape
        print(f"{name:<9} shape={shape} crs={crs} nodata={r.nodata}")
        print(f"          transform={tuple(round(v, 6) for v in transform.to_gdal())}")
    arr = data.load_tif(path, kind="no2" if name == "no2" else "guide")
    arrays[name] = arr
    print(f"          NaN={np.isnan(arr).mean() * 100:.1f}%  "
          f"min={np.nanmin(arr):.3g}  max={np.nanmax(arr):.3g}")

    if ref is None:
        ref = (shape, np.array(transform.to_gdal()), str(crs))
    else:
        same = (shape == ref[0]
                and np.allclose(transform.to_gdal(), ref[1], atol=1e-9)
                and str(crs) == ref[2])
        if not same:
            grids_ok = False
            print(f"   !!! {name} is NOT on the same grid as the NO2 map")

print("\n>>> Grids aligned:", "YES" if grids_ok else "NO (fix this before going further)")
if not grids_ok:
    sys.exit(1)

# ---------------------------------------------------------------- 4. figure
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, len(arrays), figsize=(4 * len(arrays), 3.6))
for ax, (name, arr) in zip(axes, arrays.items()):
    lo, hi = np.nanpercentile(arr, [1, 99])
    im = ax.imshow(arr, vmin=lo, vmax=hi, cmap="viridis")
    ax.set_title(name)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.7)
fig.savefig(OUT_DIR / "check_guides.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {OUT_DIR / 'check_guides.png'} "
      f"(check that the road network is at the right place and not flipped)")

# ---------------------------------------------------------------- 5. mini baseline test
FACTOR = 4
hr_full = arrays["no2"]
print(f"\n== Mini baseline test (x{FACTOR}) ==")
print(f"{'guide':<10}{'method':<17}{'RMSE':>11}{'MAE':>11}{'consist.':>11}{'gradcorr':>10}")
cubic_done = False
for guide_name in ["dem", "ghsl", "osm_dens", "osm_dist"]:
    guide = data.load_guide(FILES[guide_name], normalize=False)
    lr, hr, g = data.make_synthetic_case(hr_full, guide, FACTOR)
    preds = {
        "cubic": baselines.cubic(lr, FACTOR),
        "joint_bilateral": baselines.joint_bilateral(lr, g, FACTOR),
        "guided_filter": baselines.guided_filter(lr, g, FACTOR),
    }
    for method, pred in preds.items():
        if method == "cubic":
            if cubic_done:
                continue                 # cubic does not use the guide: print it once
            cubic_done = True
            label = "(none)"
        else:
            label = guide_name
        # gradient correlation is meaningless for a distance map (constant gradient norm)
        gc = "n/a" if guide_name == "osm_dist" else f"{metrics.gradient_correlation(pred, g):.2f}"
        print(f"{label:<10}{method:<17}{metrics.rmse(pred, hr):>11.2e}{metrics.mae(pred, hr):>11.2e}"
              f"{metrics.consistency_error(pred, lr, FACTOR):>11.2e}{gc:>10}")

print("\nIf the guided RMSE is about the same as (or worse than) cubic for every guide, the guides "
      "bring nothing (or the ground truth is already very smooth).")