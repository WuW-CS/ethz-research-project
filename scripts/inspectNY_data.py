"""
inspect_data_ny.py - quick inspection of single-band GeoTIFFs: NO2, DEM or GHSL (NYC / Long Island Sound).

What the script does
--------------------
1. Opens one or several GeoTIFFs exported from Google Earth Engine on the NY grid.
2. Prints raster information: dimensions, coordinate reference system, extent.
3. Prints statistics (min, max, mean, number of valid pixels). For NO2 the values are also given
   in 1e15 molec/cm2 so they can be compared with Judd et al. (2020).
4. Draws a map of the NYC / Long Island Sound area with a few landmark cities and the nine
   LISTOS Pandora sites, and saves it as a PNG.

Usage
-----
    python inspect_data_ny.py --kind no2 --file /scratch4/willwu/datas/NY/20180719T171554_20180725T185631.tif
    python inspect_data_ny.py --kind no2 --file /scratch4/willwu/datas/NY/*.tif       # several files at once
    python inspect_data_ny.py --kind no2 --file /scratch4/willwu/datas/NY/L3_NO2_mean/ny_no2_mean_test.tif
    # optional: --out my_figure.png (single file only), --no-pandora (hide the Pandora sites)

Dependencies: rioxarray, numpy, matplotlib, cartopy

Convention for "no-data" pixels
-------------------------------
NaN = no data. It depends on the kind whether 0 also means "no data":
- NO2 : exact 0 = empty image exported as 0 (a field that is exactly zero everywhere is not
        physical) -> converted to NaN.
- DEM : 0 m is a valid altitude (sea level, very common here) -> kept.
- GHSL: 0 means "nothing built here" -> kept. Converting it to NaN would erase the information.

About the file names
--------------------
GEE names the L3 images <start>_<second timestamp>.tif. The first timestamp (e.g. 20180719T171554)
is the UTC start time of the orbit data, which is useful to check the overpass time against the
airborne flights (S5P passes over NYC between about 17:00 and 19:00 UTC). The second timestamp is
several days later in your files, so it does not look like the end of the sensing time; do not use
it as the acquisition time. Some days have two files (e.g. 24 Aug: 15:57 and 17:38 UTC), which are
two different orbits: check which one actually covers the flight area.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")          # headless backend: required on a server (SSH)

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rioxarray               # noqa: F401  (adds .rio to xarray objects and provides open_rasterio)

# --------------------------------------------------------------------------- parameters
HERE = os.path.dirname(os.path.abspath(__file__))

# 1 mol/m2 = 6.022e23 molec / 1e4 cm2 = 6.022e19 molec/cm2  ->  to 1e15 molec/cm2:
MOL_M2_TO_1E15_MOLEC_CM2 = 6.022e19 / 1e15

# Landmark cities: name -> (longitude, latitude) in degrees (EPSG:4326)
CITIES = {
    "New York City": (-74.0060, 40.7128),
    "Newark": (-74.1724, 40.7357),
    "Stamford": (-73.5387, 41.0534),
    "Bridgeport": (-73.1952, 41.1792),
}

# LISTOS Pandora sites (coordinates from Table 4 of Judd et al., 2020):
# name -> (longitude, latitude, label offset in lon, label offset in lat)
PANDORA = {
    "QueensNY": (-73.8215, 40.7361, 0.04, -0.07),
    "BronxNY": (-73.8781, 40.8679, -0.02, 0.04),
    "BayonneNJ": (-74.1261, 40.6703, -0.04, -0.07),
    "RutgersNJ": (-74.4294, 40.4622, 0.04, -0.02),
    "FlaxPondNY": (-73.1402, 40.9635, 0.04, 0.02),
    "WestportCT": (-73.3367, 41.1183, 0.04, 0.02),
    "NewHavenCT": (-72.9029, 41.3014, 0.04, 0.02),
    "BranfordCT": (-72.7604, 41.2420, 0.04, -0.06),
    "MadisonCT": (-72.5533, 41.2568, -0.02, 0.04),
}

# Display settings per kind of raster.
#   zero_is_nodata : convert exact 0 to NaN (see docstring)
#   vmin / vmax    : color limits. None = automatic; a number = fixed; "pNN" = NN-th percentile
#   units / label  : used in the terminal and on the colorbar
KINDS = {
    "no2": dict(
        title="TROPOMI NO₂ Tropospheric Column - NYC / Long Island Sound",
        label="Tropospheric NO₂ Column Density (mol/m²)",
        units="mol/m²", cmap="viridis", zero_is_nodata=True, vmin=0, vmax="p98", fmt=".2e",
    ),
    "dem": dict(
        title="Copernicus DEM GLO-30 (mean per ~1 km cell) - NYC / Long Island Sound",
        label="Elevation (m)",
        units="m", cmap="terrain", zero_is_nodata=False, vmin=None, vmax=None, fmt=".0f",
    ),
    "ghsl": dict(
        title="GHSL built-up volume (mean per ~1 km cell) - NYC / Long Island Sound",
        label="Built-up volume (m³, mean per 100 m pixel)",   # check the unit on the GEE dataset page
        units="m³", cmap="magma", zero_is_nodata=False, vmin=0, vmax="p99", fmt=".2e",
    ),
}


def resolve_limit(value, data):
    """Turn a limit setting (None, number or 'pNN') into a number or None."""
    if value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and value.startswith("p"):
        return float(np.nanpercentile(data, float(value[1:])))
    raise ValueError(f"Unknown limit setting: {value!r}")


def start_time_from_name(path):
    """Return 'YYYY-MM-DD HH:MM UTC' parsed from a GEE file name, or None (e.g. for the mean image)."""
    m = re.search(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})\d{2}_\d{8}T\d{6}", os.path.basename(path))
    if m is None:
        return None
    y, mo, d, h, mi = m.groups()
    return f"{y}-{mo}-{d} {h}:{mi} UTC"


def inspect(path, kind, out, show_pandora=True):
    """Print the raster info + statistics of `path` and save a map to `out`."""
    cfg = KINDS[kind]

    # ----------------------------------------------------------------------- 1. reading
    ds = rioxarray.open_rasterio(path)                    # DataArray of shape (band, y, x)

    print("=== RASTER INFORMATION ===")
    print(f"File : {path}")
    t0 = start_time_from_name(path)
    if t0:
        print(f"Start time (from file name) : {t0}")
    print(f"Dimensions (bands, y, x) : {ds.shape}")
    print(f"Coordinate Reference System (CRS) : {ds.rio.crs}")   # EPSG:4326 = lon/lat in degrees
    print(f"Spatial bounds : {ds.rio.bounds()}")                 # (lon_min, lat_min, lon_max, lat_max)
    res = ds.rio.resolution()
    print(f"Pixel size (deg) : {abs(res[0]):.5f} x {abs(res[1]):.5f}")

    # First (and only) band -> 2D array (y, x). float so that it can hold NaN.
    data = ds.values[0].astype(float)

    # Declared GeoTIFF no-data value (if any) -> NaN.
    nodata = ds.rio.nodata
    if nodata is not None and np.isfinite(nodata):
        data[data == nodata] = np.nan

    if cfg["zero_is_nodata"]:
        data[data == 0] = np.nan

    # ----------------------------------------------------------------------- 2. statistics
    print(f"\n=== {kind.upper()} STATISTICS ===")
    n_valid = int(np.isfinite(data).sum())
    print(f"Valid pixels : {n_valid} / {data.size} ({n_valid / data.size:.1%})")

    if n_valid == 0:
        print("No valid pixel: nothing to plot but an empty map will be saved.")
    else:
        f = cfg["fmt"]
        print(f"Min     : {np.nanmin(data):{f}} {cfg['units']}")
        print(f"Max     : {np.nanmax(data):{f}} {cfg['units']}")
        print(f"Mean    : {np.nanmean(data):{f}} {cfg['units']}")
        if kind == "no2":
            c = MOL_M2_TO_1E15_MOLEC_CM2
            print(f"In 1e15 molec/cm2 (paper units): min {np.nanmin(data) * c:.2f} | "
                  f"max {np.nanmax(data) * c:.2f} | mean {np.nanmean(data) * c:.2f}")

    # ----------------------------------------------------------------------- 3. map
    # imshow expects [lon_min, lon_max, lat_min, lat_max] while rio.bounds() returns
    # (lon_min, lat_min, lon_max, lat_max) -> hence the reordering.
    bounds = ds.rio.bounds()
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())      # plain lon/lat projection, same as the data
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    vmin = resolve_limit(cfg["vmin"], data) if n_valid > 0 else None
    vmax = resolve_limit(cfg["vmax"], data) if n_valid > 0 else None

    img_plot = ax.imshow(
        data,
        origin="upper",            # row 0 of the array = north (raster convention)
        extent=extent,             # geographic coordinates of the image edges
        transform=ccrs.PlateCarree(),
        cmap=cfg["cmap"],
        vmin=vmin,
        vmax=vmax,
    )

    # 50 m coastline: the default 110 m one is too coarse for Long Island and the Sound.
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linestyle=":", edgecolor="black", linewidth=1)

    # Landmark cities: a marker + a label. New York City is a red star, the others orange dots.
    for city_name, (lon, lat) in CITIES.items():
        is_nyc = city_name == "New York City"
        ax.plot(
            lon, lat,
            marker="*" if is_nyc else "o",
            color="red" if is_nyc else "orange",
            markeredgecolor="black", markeredgewidth=0.5,   # stays visible on any colormap
            markersize=12 if is_nyc else 6,
            transform=ccrs.PlateCarree(),
        )
        ax.text(
            lon + 0.03, lat + 0.03, city_name,     # small offset so the label does not cover the marker
            transform=ccrs.PlateCarree(),
            fontsize=12 if is_nyc else 10,
            fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    # Pandora sites: small white triangles with a short label (several are very close to each other).
    if show_pandora:
        for name, (lon, lat, dx, dy) in PANDORA.items():
            ax.plot(lon, lat, marker="^", color="white", markeredgecolor="black",
                    markeredgewidth=0.6, markersize=7, transform=ccrs.PlateCarree())
            ax.text(
                lon + dx, lat + dy, name,
                transform=ccrs.PlateCarree(), fontsize=7, color="black",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"),
            )

    plt.colorbar(img_plot, ax=ax, orientation="vertical", pad=0.02, label=cfg["label"])
    title = cfg["title"] + (f"\nstart: {t0}" if t0 else "")
    plt.title(title, fontsize=14, fontweight="bold")

    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()                    # free the figure memory
    print(f"\nMap saved at '{out}'\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect NO2 / DEM / GHSL GeoTIFFs (NYC) and plot them.")
    parser.add_argument("--kind", choices=sorted(KINDS), required=True, help="type of raster")
    parser.add_argument("--file", required=True, nargs="+", help="path(s) to the .tif file(s)")
    parser.add_argument("--out", default=None, help="output PNG (only with a single file)")
    parser.add_argument("--no-pandora", action="store_true", help="do not draw the Pandora sites")
    args = parser.parse_args()

    if args.out and len(args.file) > 1:
        parser.error("--out can only be used with a single --file")

    for path in args.file:
        stem = os.path.splitext(os.path.basename(path))[0]
        out = args.out or os.path.join(HERE, f"{args.kind}_ny_{stem}_preview.png")
        inspect(path, args.kind, out, show_pandora=not args.no_pandora)


if __name__ == "__main__":
    main()