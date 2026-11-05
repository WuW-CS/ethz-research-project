"""
run_baselines.py - simple synthetic test of the baselines on one NO2 map.

For each guide (DEM, GHSL) and each factor:
    lr   = block_average(hr)                    (fake coarse observation made from the real map)
    pred = method(lr, guide, factor)            (cubic, joint bilateral, guided filter)
    compare pred with hr                        (RMSE, MAE, consistency, gradient correlation)

Run from anywhere:  python scripts/run_baselines.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless backend: required on a server (SSH)
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import baselines, data, metrics    

# ----------------------------------------------------------------------------- settings
DATAS = Path("/scratch4/willwu/datas/madrid")
NO2_FILE = DATAS / "L3_NO2_mean" / "madrid_no2_mean.tif"
GUIDE_FILES = {                                   # first file matching each name
    "dem": sorted((DATAS / "guides").glob("*dem*.tif"))[0],
    "ghsl": sorted((DATAS / "guides").glob("*ghsl*.tif"))[0],
}
FACTORS = [2, 4, 8]
OUT_DIR = ROOT / "outputs" / "figures"

# ----------------------------------------------------------------------------- data
hr_full = data.load_tif(NO2_FILE)                 # the "truth" (H, W)
# To use one day instead of the monthly mean:
# hr_full = data.load_daily_maps(DATAS / "L3_NO2", min_valid=0.9)["20250903"]

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- experiment
print(f"{'guide':<6}{'x':>3} {'method':<17}{'RMSE':>11}{'MAE':>11}{'consist.':>11}{'gradcorr':>10}")

for guide_name, guide_path in GUIDE_FILES.items():
    guide = data.load_guide(guide_path, normalize=False)

    for factor in FACTORS:
        lr, hr, g = data.make_synthetic_case(hr_full, guide, factor)

        preds = {
            "cubic": baselines.cubic(lr, factor),
            "joint_bilateral": baselines.joint_bilateral(lr, g, factor),
            "guided_filter": baselines.guided_filter(lr, g, factor),
        }

        for method, pred in preds.items():
            print(f"{guide_name:<6}{factor:>3} {method:<17}"
                  f"{metrics.rmse(pred, hr):>11.2e}{metrics.mae(pred, hr):>11.2e}"
                  f"{metrics.consistency_error(pred, lr, factor):>11.2e}"
                  f"{metrics.gradient_correlation(pred, g):>10.2f}")

        # one picture: truth | lr | the three methods
        vmax = float(np.nanpercentile(hr, 99))
        lr_show = np.repeat(np.repeat(lr, factor, axis=0), factor, axis=1)
        panels = {"truth (hr)": hr, f"lr (x{factor})": lr_show, **preds}
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6))
        for ax, (title, img) in zip(axes, panels.items()):
            im = ax.imshow(img, vmin=0, vmax=vmax, cmap="viridis")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.colorbar(im, ax=axes, shrink=0.8, label="NO2 (mol/m2)")
        fig.suptitle(f"guide: {guide_name} - factor {factor}")
        fig.savefig(OUT_DIR / f"baselines_{guide_name}_x{factor}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

print(f"\nFigures saved in {OUT_DIR}")