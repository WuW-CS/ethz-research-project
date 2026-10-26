"""
inspect_data.py - quick inspection of a single-band GeoTIFF: NO2, DEM or GHSL.

What the script does
--------------------
1. Opens a GeoTIFF exported from Google Earth Engine on the Madrid grid.
2. Prints raster information: dimensions, coordinate reference system, extent.
3. Prints statistics (min, max, mean, number of valid pixels).
4. Draws a map of the Madrid area with a few landmark cities and saves it as a PNG.

Usage
-----
    python inspect_data.py --kind no2  --file /scratch4/willwu/datas/madrid/L3_NO2/<file>.tif
    python inspect_data.py --kind dem  --file /scratch4/willwu/datas/madrid/guides/<dem>.tif
    python inspect_data.py --kind ghsl --file /scratch4/willwu/datas/madrid/guides/<ghsl>.tif
    # optional: --out my_figure.png   (default: <kind>_madrid_preview.png in this folder)

Dependencies: rioxarray, numpy, matplotlib, cartopy

Convention for "no-data" pixels
-------------------------------
NaN = no data. It depends on the kind whether 0 also means "no data":
- NO2 : exact 0 = empty image exported as 0 (a field that is exactly zero everywhere is not
        physical) -> converted to NaN.
- DEM : 0 m is a valid altitude -> kept.
- GHSL: 0 means "nothing built here" (very common in the countryside) -> kept. Converting it
        to NaN would erase the information.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")          # headless backend: required on a server (SSH)

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rioxarray               # noqa: F401  (adds .rio to xarray objects and provides open_rasterio)

# --------------------------------------------------------------------------- parameters
HERE = os.path.dirname(os.path.abspath(__file__))

# Landmark cities: name -> (longitude, latitude) in degrees (EPSG:4326)
CITIES = {
    "Madrid": (-3.7038, 40.4168),
    "Getafe": (-3.7333, 40.3083),
    "Móstoles": (-3.8648, 40.3235),
    "Alcalá de Henares": (-3.3639, 40.4819),
    "Alcobendas": (-3.6333, 40.5333),
}

# Display settings per kind of raster.
#   zero_is_nodata : convert exact 0 to NaN (see docstring)
#   vmin / vmax    : color limits. None = automatic; a number = fixed; "pNN" = NN-th percentile
#   units / label  : used in the terminal and on the colorbar
KINDS = {
    "no2": dict(
        title="TROPOMI NO₂ Tropospheric Column - Greater Madrid Area",
        label="Tropospheric NO₂ Column Density (mol/m²)",
        units="mol/m²", cmap="viridis", zero_is_nodata=True, vmin=0, vmax="p98", fmt=".2e",
    ),
    "dem": dict(
        title="Copernicus DEM GLO-30 (mean per ~1 km cell) - Greater Madrid Area",
        label="Elevation (m)",
        units="m", cmap="terrain", zero_is_nodata=False, vmin=None, vmax=None, fmt=".0f",
    ),
    "ghsl": dict(
        title="GHSL built-up volume (mean per ~1 km cell) - Greater Madrid Area",
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


def inspect(path, kind, out):
    """Print the raster info + statistics of `path` and save a map to `out`."""
    cfg = KINDS[kind]

    # ----------------------------------------------------------------------- 1. reading
    ds = rioxarray.open_rasterio(path)                    # DataArray of shape (band, y, x)

    print("=== RASTER INFORMATION ===")
    print(f"File : {path}")
    print(f"Dimensions (bands, y, x) : {ds.shape}")              # e.g. (1, 182, 235)
    print(f"Coordinate Reference System (CRS) : {ds.rio.crs}")   # e.g. EPSG:4326 = lon/lat in degrees
    print(f"Spatial bounds : {ds.rio.bounds()}")                 # (lon_min, lat_min, lon_max, lat_max)

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

    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black", linewidth=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=1)

    # Landmark cities: a marker + a label. Madrid is a red star, the others orange dots.
    for city_name, (lon, lat) in CITIES.items():
        is_madrid = city_name == "Madrid"
        ax.plot(
            lon, lat,
            marker="*" if is_madrid else "o",
            color="red" if is_madrid else "orange",
            markeredgecolor="black", markeredgewidth=0.5,   # stays visible on any colormap
            markersize=12 if is_madrid else 6,
            transform=ccrs.PlateCarree(),
        )
        ax.text(
            lon + 0.03, lat + 0.03, city_name,     # small offset so the label does not cover the marker
            transform=ccrs.PlateCarree(),
            fontsize=12 if is_madrid else 10,
            fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    plt.colorbar(img_plot, ax=ax, orientation="vertical", pad=0.02, label=cfg["label"])
    plt.title(cfg["title"], fontsize=14, fontweight="bold")

    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()                    # free the figure memory
    print(f"\nMap saved at '{out}'")


def main():
    parser = argparse.ArgumentParser(description="Inspect a NO2 / DEM / GHSL GeoTIFF and plot it.")
    parser.add_argument("--kind", choices=sorted(KINDS), required=True, help="type of raster")
    parser.add_argument("--file", required=True, help="path to the .tif file")
    parser.add_argument("--out", default=None, help="output PNG (default: <kind>_madrid_preview.png)")
    args = parser.parse_args()

    out = args.out or os.path.join(HERE, f"{args.kind}_madrid_preview.png")
    inspect(args.file, args.kind, out)


if __name__ == "__main__":
    main()