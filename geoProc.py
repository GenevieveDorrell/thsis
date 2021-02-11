import rasterio as rs
import geopandas as gpd
from rasterio.plot import show
from rasterio.enums import ColorInterp
import numpy

#file paths for bigger fire
sev1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_20140706_dnbr6.tif"
rgb1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_l8_refl.tif"

#file paths for smaller fire
sev2_fp = "data\\2013\or4285712358520130726\or4285712358520130726_20130703_20140706_dnbr6.tif"
rgb2_fp = "data\\2013\or4285712358520130726\or4285712358520130726_20130703_l8_refl.tif"


#open tiffs
def get_pixel_array(fp):
    img = rs.open(fp)
    array = img.read()
    img.close()
    return array

def get_numpy_from_tiff(severity_fp, rgb_fp):
    severity_array = get_pixel_array(severity_fp)
    rgb_array = get_pixel_array(rgb_fp)
    sev = []
    val = []
    for i in range(len(severity_array)):
        for j in range(len(severity_array[0])):
            for k in range(len(severity_array[0][0])):
                if severity_array[i][j][k] != 0:
                    sev.append(severity_array[i][j][k]-1)
                    vals = []
                    for l in range(len(rgb_array)):
                        vals.append(rgb_array[l][j][k])
                    val.append(vals)
    severity = numpy.array(sev)
    rgb = numpy.array(val)

    return severity, rgb











