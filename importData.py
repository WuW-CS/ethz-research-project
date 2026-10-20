import ee
import geemap

ee.Initialize()

region = ee.Geometry.Rectangle([5.5, 45.5, 10.5, 48.0])

# img = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
#        .select("tropospheric_NO2_column_number_density")
#        .filterDate("2025-09-01", "2025-09-16")
#        .mean()
#       )

# out_dir = "/scratch4/willwu/datas/test/no2_switzerland.tif"
# geemap.ee_export_image(img, filename=out_dir, scale=1113.2, region=region, crs="EPSG:4326")
collection = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
              .select("tropospheric_NO2_column_number_density")
              .filterDate("2025-09-01", "2025-09-05")
              .filterBounds(region)
              )

geemap.ee_export_image_collection(
       collection,
       out_dir = "/scratch4/willwu/datas/test/",
       region = region,
       crs ="EPSG:4326"
)