import rasterio as rs

from typing import List
import numpy
import time


from pyproj import Transformer, CRS


#
lossyear = "data\\2013\Hansen_GFC2013_lossyear_50N_130W.tif"
treecover = "data\\2013\Hansen_GFC-2019-v1.7_treecover2000_50N_130W.tif"

#file paths for bigger fire
sev1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_20140706_dnbr6.tif"
rgb1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_l8_refl.tif"

#file paths for smaller fire
sev2_fp = "data\\2013\or4285712358520130726\or4285712358520130726_20130703_20140706_dnbr6.tif"
rgb2_fp = "data\\2013\or4285712358520130726\or4285712358520130726_20130703_l8_refl.tif"


#open tiffs
def open_tiffs(fps: List[str]):
    """
    opens all the tiffs at the file locations in the input list
    and returens their coresponding rasterio dataset object and 
    an array is pixel values
    """
    obj = []
    arrays = []
    for file_path in fps:
        try:
            img = rs.open(file_path)
            array = img.read()
            obj.append(img)
            arrays.append(array)
        except:
            print("cannot open file at " + file_path)
    return arrays, obj

def get_arrays_from_dnbr6(severity_array: "numpy", rgb_array: "numpy"):
    sev = []
    rgb = []
    row_col = []
    for i in range(len(severity_array[0])):
        for j in range(len(severity_array[0][0])):
            if severity_array[0][i][j] != 0 and severity_array[0][i][j] < 6:
                sev.append(severity_array[0][i][j]-1)
                row_col.append((i,j))
                vals = []
                for l in range(len(rgb_array)):
                    vals.append(rgb_array[l][i][j])
                rgb.append(vals)

    return sev, rgb, row_col



def declare_transformation_objs(sev_obj, tiff_objs: List):
    transformations = []
    incrs = CRS(sev_obj.crs.to_string())
    for tiff in tiff_objs:
        outcrs = CRS(tiff.crs.to_string())
        transformer = Transformer.from_crs(incrs, outcrs)
        transformations.append(transformer)
    return transformations 

def close_tiffs(tiff_objs: List):
    for tiff in tiff_objs:
        tiff.close()
    


def get_numpy_from_tiffs(severity_fps: List[str], rgbs_fps: List[str], data_fps: List[str]):
    """
    puts lables and pixel data from tiffs into numpy arrays
    """
    #declare stuff once
    severity_data, sev_objs  = open_tiffs(severity_fps)
    rgb_data, throw_away = open_tiffs(rgbs_fps)
    tiff_data, tiff_objs = open_tiffs(data_fps)

    lables = []
    data = []

    #get data based on current sev location and transformations
    for i in range(len(sev_objs)):
        sev_obj = sev_objs[i]
        #loop through once to get row_col values and rgbs
        sev, rgbs, row_col = get_arrays_from_dnbr6(severity_data[i], rgb_data[i])
        transformers = declare_transformation_objs(sev_obj, tiff_objs)
        for j in range(len(row_col)):
            row, col = row_col[j]
            lon, lat = sev_obj.xy(row,col)
            for tiff in range(len(tiff_objs)):
                lon2, lat2 = transformers[tiff].transform(lon, lat)
                row2, col2 = tiff_objs[tiff].index(lat2, lon2)
                for k in range(len(tiff_data[tiff])):
                    rgbs[j].append(tiff_data[tiff][k][row2][col2])
           
        lables += sev
        data += rgbs

    close_tiffs(sev_objs)
    close_tiffs(throw_away)
    close_tiffs(tiff_objs)
    print("calculating test cases")
    test_lables = []
    test_data = []
    catgories = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for i in range(len(lables)-250):
        if catgories[lables[i]] < 50:
            catgories[lables[i]] += 1
            test_data.append(data[i])
            test_lables.append(lables[i])
            data.pop(i)
            lables.pop(i)
            i -= 1
            


    severity = numpy.array(lables)
    all_data = numpy.array(data)
    test_severity = numpy.array(test_lables)
    test_all_data = numpy.array(test_data)

    print(catgories)

    return severity, test_severity, all_data, test_all_data



if __name__ == "__main__":


    lables, data = get_numpy_from_tiffs([sev1_fp, sev2_fp],[rgb1_fp, rgb2_fp], [treecover, lossyear])
    print(data.shape)
    print(lables.shape)







