"""
inspect_data.py - quick inspection of a tropospheric NO2 GeoTIFF (TROPOMI / Sentinel-5P).

What the script does
--------------------
1. Opens a GeoTIFF exported from Google Earth Engine (one image = one satellite overpass).
2. Prints raster information: dimensions, coordinate reference system, extent.
3. Computes NO2 statistics (min, max, mean, number of valid pixels).
4. Draws a map of the Madrid area with a few landmark cities and saves it as a PNG.

Input    : a .tif file (1 band: tropospheric_NO2_column_number_density, in mol/m2)
Output   : statistics in the terminal + a PNG file

Dependencies: rioxarray, numpy, matplotlib, cartopy

Convention for "no-data" pixels
-------------------------------
A pixel has no data (cloud, outside the satellite swath, quality filter) -> NaN.
But some exports write 0 instead of NaN for completely empty images.
A NO2 field that is exactly zero everywhere is not physical, so we treat 0 as "no data".
"""

import matplotlib
matplotlib.use("Agg")          # headless backend: required on a server (SSH)

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rioxarray               # noqa: F401  (adds .rio to xarray objects and provides open_rasterio)

# --------------------------------------------------------------------------- parameters
FILE_PATH = "/scratch4/willwu/datas/madrid/20250905T112639_20250907T035307.tif"
OUTPUT_PREVIEW = "/scratch4/willwu/ethz-research-project/no2_madrid_preview.png"

# Landmark cities: name -> (longitude, latitude) in degrees (EPSG:4326)
CITIES = {
    "Madrid": (-3.7038, 40.4168),
    "Getafe": (-3.7333, 40.3083),
    "Móstoles": (-3.8648, 40.3235),
    "Alcalá de Henares": (-3.3639, 40.4819),
    "Alcobendas": (-3.6333, 40.5333),
}


# --------------------------------------------------------------------------- 1. reading
ds = rioxarray.open_rasterio(FILE_PATH)   # DataArray of shape (band, y, x)

print("=== RASTER INFORMATION ===")
print(f"Dimensions (bands, y, x) : {ds.shape}")             # e.g. (1, 182, 235)
print(f"Coordinate Reference System (CRS) : {ds.rio.crs}")  # e.g. EPSG:4326 = lon/lat in degrees
print(f"Spatial bounds : {ds.rio.bounds()}")                # (lon_min, lat_min, lon_max, lat_max)

# Take the first (and only) band -> 2D array (y, x). Cast to float so it can hold NaN.
data = ds.values[0].astype(float)

# CHANGE: exact 0 = empty image exported as 0 -> convert it to NaN.
data[data == 0] = np.nan


# --------------------------------------------------------------------------- 2. statistics
print("\n=== TROPOSPHERIC NO2 STATISTICS ===")
n_valid = int(np.isfinite(data).sum())
print(f"Valid pixels : {n_valid} / {data.size} ({n_valid / data.size:.1%})")

if n_valid == 0:
    # nanmin/nanmax/nanmean warn and return NaN when everything is NaN.
    print("No valid pixel: image has no data over this area (overpass elsewhere, or clouds).")
else:
    # The nan* functions ignore NaN, unlike min/max/mean.
    print(f"Min     : {np.nanmin(data):.2e} mol/m²")
    print(f"Max     : {np.nanmax(data):.2e} mol/m²")
    print(f"Mean    : {np.nanmean(data):.2e} mol/m²")


# --------------------------------------------------------------------------- 3. map
# CHANGE: negative values are no longer masked. TROPOMI NO2 can have small negative values
# due to measurement noise; removing them would bias the statistics.
data_plot = data

# Image extent to position the map: imshow expects [lon_min, lon_max, lat_min, lat_max]
# while rio.bounds() returns (lon_min, lat_min, lon_max, lat_max) -> hence the reordering.
bounds = ds.rio.bounds()
extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())       # plain lon/lat projection, same as the data
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Color scale: from 0 to the 98th percentile, so a few extreme values do not
# squash everything else. If the image is empty, let matplotlib choose.
vmax = float(np.nanpercentile(data_plot, 98)) if n_valid > 0 else None

img_plot = ax.imshow(
    data_plot,
    origin="upper",            # row 0 of the array = north (raster convention)
    extent=extent,             # geographic coordinates of the image edges
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    vmin=0 if n_valid > 0 else None,
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

plt.colorbar(img_plot, ax=ax, orientation="vertical", pad=0.02,
             label="Tropospheric NO₂ Column Density (mol/m²)")
plt.title("TROPOMI NO₂ Tropospheric Column - Greater Madrid Area", fontsize=14, fontweight="bold")

plt.savefig(OUTPUT_PREVIEW, dpi=300, bbox_inches="tight")
plt.close()                    # free the figure memory

print(f"\nMap saved at '{OUTPUT_PREVIEW}'")