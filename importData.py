import ee
import geemap

# Initialize Earth Engine
ee.Initialize()

# Bounding box for Madrid (approx. 200x200 km)
# Format: [Min Longitude, Min Latitude, Max Longitude, Max Latitude]
region = ee.Geometry.Rectangle([-4.87,39.51,-2.53,41.31])

band ="tropospheric_NO2_column_number_density"

def add_valid_count(img):
       n= img.reduceRegion(ee.Reducer.count(), region, 1113.2,maxPixels=1e9).get(band)
       return img.set("n_valid",n)

# Load TROPOMI NO2 Level-3 dataset
collection = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
              .select(band)
              .filterDate("2025-09-01", "2025-09-16")
              .filterBounds(region)
              .map(add_valid_count)                     
              .filter(ee.Filter.gt("n_valid",2000))
              )



# Batch export to local directory
geemap.ee_export_image_collection(
       collection,
       out_dir = "/scratch4/willwu/datas/madrid/",
       scale = 1113.2,
       region = region,
       crs ="EPSG:4326"
)