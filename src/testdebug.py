import glob, collections, numpy as np, rioxarray

days = collections.defaultdict(list)
for f in sorted(glob.glob("/scratch4/willwu/datas/madrid/*.tif")):
    d = rioxarray.open_rasterio(f).values[0].astype(float)
    d[d == 0] = np.nan                     # 0 = image vide, pas du NO2
    if np.isfinite(d).mean() > 0.05:       # ignore les images (quasi) vides
        days[f.split("/")[-1][:8]].append(d)

daily = {k: np.nanmean(v, axis=0) for k, v in days.items()}   # un champ par jour
for k, v in sorted(daily.items()):
    print(k, f"valides: {np.isfinite(v).mean():.0%}", f"moyenne: {np.nanmean(v):.2e}")