import fiona
import rasterio
import rasterio.mask

shape = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_20140706_burn_bndy.shp"
file = "data\\2013\Hansen_GFC2013_lossyear_50N_130W.tif"

import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
"""
#dst_crs = fiona.open(shape, "r")#'EPSG:4326'
with fiona.open(shape) as shapefile:
    print(shapefile.meta['crs_wkt'])
with rasterio.open(file) as src:
    print(src.meta)

with rasterio.open(file) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open('/tmp/RGB.byte.wgs84.tif', 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

"""

with fiona.open(shape, "r") as shapefile:
    print(shapefile)
    shapes = [feature["geometry"] for feature in shapefile]
    #print(shapes)

with rasterio.open(file) as src:
    print(src.meta)
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=False)
    out_meta = src.meta

out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

with rasterio.open("RGB.masked.tif", "w", **out_meta) as dest:
    dest.write(out_image)
