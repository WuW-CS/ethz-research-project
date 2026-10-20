import xarray as xr
import rioxarray  
import numpy as np
import matplotlib.pyplot as plt


file_path = "/scratch4/willwu/datas/test/no2_switzerland.tif"


ds = rioxarray.open_rasterio(file_path)

print("=== INFORMATIONS===")
print(f"Dimensions (bands, y, x) : {ds.shape}")
print(f"Coordinate System(CRS) : {ds.rio.crs}")
print(f"Spatial bounds (bounds) : {ds.rio.bounds()}")
print(f"Data type : {ds.dtype}")

# 2. Extract the first band (NO2 density) and calculate some values
data = ds.values[0] # La première bande
print("\n=== STATISTICAL DATAS NO2 ===")
print(f"Min : {np.nanmin(data):.2e} mol/m²")
print(f"Max : {np.nanmax(data):.2e} mol/m²")
print(f"Mean    : {np.nanmean(data):.2e} mol/m²")
print(f"Number of NaN : {np.isnan(data).sum()} / {data.size}")

# 3. Save a png
plt.figure(figsize=(10, 6))

data_plot = np.where(data < 0, np.nan, data)
plt.imshow(data_plot, cmap='viridis')
plt.colorbar(label='NO2 Column Density (mol/m²)')
plt.title("TROPOMI NO2 Tropospheric Column - Switzerland")
plt.xlabel("X (Pixels)")
plt.ylabel("Y (Pixels)")
plt.savefig("/scratch4/willwu/ethz-research-project/no2_preview.png", dpi=300)
