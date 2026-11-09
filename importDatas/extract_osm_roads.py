import os
import numpy as np
import rioxarray
from affine import Affine
from pyrosm import OSM
from rasterio import features
from scipy.ndimage import distance_transform_edt

osm_pbf_path = "/scratch4/willwu/datas/madrid/guides/spain-latest.osm.pbf"
out_dir = "/scratch4/willwu/datas/madrid/guides/"
template_path = out_dir + "dem_glo30_2024_1.tif"
K = 10  # thin sub-sampling (each cell ~1.1 km -> 10x10 sub-pixel)

# 1. template + bbox derived from raster
tpl = rioxarray.open_rasterio(template_path).squeeze(drop=True)
H, W = tpl.shape
minx, miny, maxx, maxy = tpl.rio.bounds()
m = 0.05
bbox_osm = [minx - m, miny - m, maxx + m, maxy + m]
print("CRS gabarit:", tpl.rio.crs, "| shape:", tpl.shape, "| bbox:", bbox_osm)

# 2. Lecture OSM
edges = OSM(osm_pbf_path, bounding_box=bbox_osm).get_network(nodes=False, network_type="driving")
if edges is None or edges.empty:
    raise SystemExit("No roads found")
edges = edges.to_crs(tpl.rio.crs)
print(f"{len(edges)} segments")

# 3. Rasterisation on thin grid
fine_tr = tpl.rio.transform() * Affine.scale(1 / K, 1 / K)
fine = features.rasterize(((g, 1) for g in edges.geometry if g is not None),
                          out_shape=(H * K, W * K), transform=fine_tr,
                          fill=0, all_touched=True, dtype="uint8")

# 4a. density of road per cell
density = fine.reshape(H, K, W, K).mean(axis=(1, 3))
print("% cellules avec route:", (density > 0).mean() * 100)

# 4b. Distance (km) to the nearest road pixel, on thin grid
res_x, res_y = abs(tpl.rio.resolution()[0]), abs(tpl.rio.resolution()[1])
lat0 = (miny + maxy) / 2
dy_km = res_y / K * 111.32
dx_km = res_x / K * 111.32 * np.cos(np.deg2rad(lat0))
dist_fine = distance_transform_edt(fine == 0, sampling=(dy_km, dx_km))
dist = dist_fine.reshape(H, K, W, K).mean(axis=(1, 3))  # mean distance per cell

# 5. Export
def save(arr, name):
    out = tpl.copy(data=arr.astype("float32"))
    out.rio.to_raster(out_dir + name)
    print("saved", name)

save(np.log1p(1000 * density), "osm_roads_density_madrid.tif")
save(np.log1p(dist), "osm_roads_distance_madrid.tif")