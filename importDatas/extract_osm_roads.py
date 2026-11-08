import os
import numpy as np
from pyrosm import OSM
from geocube.api.core import make_geocube
from scipy.ndimage import distance_transform_edt
import rasterio
import rioxarray

# ==========================================
# PARAMETERS
# ==========================================
# Ensure "spain-latest.osm.pbf" is downloaded in the same directory via wget
osm_pbf_path = "/scratch4/willwu/datas/madrid/guides/spain-latest.osm.pbf"
bbox_osm = [-4.87, 39.51, -2.53, 41.31] # [min_lon, min_lat, max_lon, max_lat]

out_dir = "/scratch4/willwu/datas/madrid/guides/"
os.makedirs(out_dir, exist_ok=True)

# ==========================================
# 1. READ OSM DATA
# ==========================================
print("1. Reading OSM data for the specified bounding box...")
osm = OSM(osm_pbf_path, bounding_box=bbox_osm)

# Extract only driving roads
edges = osm.get_network(nodes=False, network_type="driving")

if edges is None or edges.empty:
    print("No roads found in the specified bounding box. Exiting.")
    exit()

print(f"2. Found {len(edges)} road segments. Preparing for rasterization...")

# ==========================================
# 2. RASTERIZE ROAD PRESENCE
# ==========================================
# Assign a constant value to represent the presence of a road
edges["road_presence"] = 1

print("3. Rasterizing roads using GeoCube...")

template_path = out_dir + "dem_glo30_2024_1.tif"
template_raster = rioxarray.open_rasterio(template_path)
cube = make_geocube(
    vector_data=edges,
    measurements=["road_presence"],
    like= template_raster, 
    fill=0 # Fill empty areas with 0
)

# ==========================================
# 3. CALCULATE DISTANCE TO NEAREST ROAD
# ==========================================
print("4. Calculating distance to nearest road (for NO2 analysis)...")
presence_array = cube["road_presence"].values

# The distance_transform_edt function calculates the distance to the nearest ZERO.
# We want the distance TO the roads. So we invert our logic:
# Roads become 0 (target), background becomes 1.
background_mask = (presence_array == 0)

# Calculate Euclidean distance (in pixels)
distance_in_pixels = distance_transform_edt(background_mask)

res_x = abs(template_raster.rio.resolution()[0])
distance_in_degrees = distance_in_pixels * res_x

# Add the distance array as a new variable in our dataset
cube["road_distance"] = (cube["road_presence"].dims, distance_in_degrees)

# ==========================================
# 4. EXPORT TO GEOTIFF
# ==========================================
out_dist_file = out_dir + "osm_roads_distance_madrid.tif"
print(f"5. Saving distance raster to {out_dist_file}...")

# Export the distance map
cube["road_distance"].rio.to_raster(out_dist_file)

print("OSM distance processing completed successfully!")