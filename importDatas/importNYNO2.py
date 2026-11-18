import ee
import geemap
import os

ee.Initialize()

# Bounding box for NYC + Long Island Sound (LISTOS domain, with margin)
# Format: [Min Lon, Min Lat, Max Lon, Max Lat]
region = ee.Geometry.Rectangle([-74.8, 40.2, -72.2, 41.6])
band = "tropospheric_NO2_column_number_density"
scale = 1113.2

# LISTOS flight days with TROPOMI coincidences available in GEE (after 10 July 2018)
flight_days = ["2018-07-19", "2018-07-20", "2018-08-06", "2018-08-15",
               "2018-08-16", "2018-08-24", "2018-08-28", "2018-08-29",
               "2018-09-06"]

def add_valid_count(img):
    n = img.reduceRegion(ee.Reducer.count(), region, scale, maxPixels=1e9).get(band)
    return img.set("n_valid", n).set("day", img.date().format("YYYY-MM-dd"))

# Load TROPOMI NO2 Level-3 dataset (flight days only)
collection = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
              .select(band)
              .filterDate("2018-07-10", "2018-09-08")
              .filterBounds(region)
              .map(add_valid_count)
              .filter(ee.Filter.inList("day", flight_days))
              .filter(ee.Filter.gt("n_valid", 2000)))

out_dir = "/scratch4/willwu/datas/NY/"
os.makedirs(out_dir, exist_ok=True)

geemap.ee_export_image_collection(
    collection,
    out_dir=out_dir,
    scale=scale,
    region=region,
    crs="EPSG:4326",
)

# Mean image over the whole period (optional test image)
image_mean = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
              .select(band)
              .filterDate("2018-07-10", "2018-09-08")
              .filterBounds(region)
              .map(add_valid_count)
              .filter(ee.Filter.gt("n_valid", 2000))
              .mean())

mean_dir = os.path.join(out_dir, "L3_NO2_mean")
os.makedirs(mean_dir, exist_ok=True)

geemap.ee_export_image(
    image_mean,
    filename=os.path.join(mean_dir, "ny_no2_mean_test.tif"),
    scale=scale,
    region=region,
    crs="EPSG:4326",
)