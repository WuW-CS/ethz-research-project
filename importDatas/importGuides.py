import ee
import geemap

ee.Initialize()

SCALE, CRS = 1113.2, "EPSG:4326"
region = ee.Geometry.Rectangle([-4.87,39.51,-2.53,41.31])


def to_grid(img):
    """Average the fine-resolution pixels within each cell of the NO2 grid."""
    return (img.reduceResolution(ee.Reducer.mean(), maxPixels=4096)
               .reproject(crs=CRS, scale=SCALE))

ghsl =to_grid(ee.Image('JRC/GHSL/P2023A/GHS_BUILT_V/2025')
        .select('built_volume_total')
       )

col = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM")
dem = to_grid(col.mosaic().setDefaultProjection(col.first().projection()))

out = "/scratch4/willwu/datas/madrid/guides/"   
geemap.ee_export_image(ghsl, filename=out + "ghsl_built_volum_total_2025.tif", scale=SCALE, region=region, crs=CRS)
geemap.ee_export_image(dem,  filename=out + "dem_2025.tif",scale=SCALE, region=region, crs=CRS)
task = ee.batch.Export.image.toDrive(
    image=dem,
    description="dem_glo30_madrid",        
    folder="gee_exports",                  
    fileNamePrefix="dem_glo30_2024_1",     
    region=region,
    scale=SCALE,
    crs=CRS,
    maxPixels=1e10,                        
    fileFormat="GeoTIFF",
)
task.start()

import time
while task.active():                       
    print(task.status()["state"])          
    time.sleep(30)
print(task.status())