import rasterio as rs
import numpy
from typing import List
#from osgeo import GDAL

#file paths for bigger fire
sev1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_20140706_dnbr6.tif"
rgb1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_l8_refl.tif"

#file paths for smaller fire
sev2_fp = "data\\2013\or4285712358520130726\or4285712358520130726_20130703_20140706_dnbr6.tif"
rgb2_fp = "data\\2013\or4285712358520130726\or4285712358520130726_20130703_l8_refl.tif"



#open tiffs
def get_pixel_arrays(fps: str):
    """
    opens all the tiffs at the file locations in the input list
    and returens their coresponding array of pixel values
    """
    opened = []
    for file_path in fps:
        try:
            img = rs.open(file_path)
            array = img.read()
            img.close()
            opened.append([array])
        except:
            print("cannot open file at " + file_path)
    return opened

def get_numpy_from_dnbr6(severity_fp: List[str], rgb_fp: List[str]):
    severity_arrays  = get_pixel_arrays(severity_fp)
    rgb_arrays = get_pixel_arrays(rgb_fp)
    sev = []
    rgb = []
    row_col = []

    for array in range(len(severity_arrays)):
        severity_array = severity_arrays[array] 
        for i in range(len(severity_array[0])):
            for j in range(len(severity_array[0][0])):
                if severity_array[0][i][j] != 0:
                    sev.append(severity_array[0][i][j]-1)
                    row_col.append((i,j))
                    vals = []
                    for l in range(len(rgb_arrays[array])):
                        vals.append(rgb_arrays[array][l][i][j])
                    rgb.append(vals)

    return sev, rgb, row_col

if __name__ == "__main__":





        






