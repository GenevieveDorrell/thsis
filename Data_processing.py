import rasterio as rs

from typing import List
import numpy
from sklearn.preprocessing import normalize
from pyproj import Transformer, CRS
from random import randint, random, randrange
import pandas as pd
from math import floor




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
        print(f"opening {file_path}")
        try:
            img = rs.open(file_path)
            array = img.read()
            obj.append(img)
            #print(img.crs.to_string())
            arrays.append(array)            
        except:
            print("cannot open file at " + file_path)
    return arrays, obj


def gen_dir_vecs(num_layers: int):
    """
    generates direction vecors to
    get layers of pixels around center pix
    """
    if num_layers == 0:
        return [[(0,0)]]
    dir_vecs = []
    for i in range(0-num_layers, num_layers+1):
        row = []
        for j in range(0-num_layers, num_layers+1):
            row.append((i,j))
        dir_vecs.append(row)
    return dir_vecs

def get_arrays_from_dnbr6(severity_array: "numpy", rgb_array: "numpy"):
    print(severity_array.shape)
    sev = {}
    rgb = {}
    for i in range(len(severity_array[0])):        
        for j in range(len(severity_array[0][0])):
            if severity_array[0][i][j] != 0 and severity_array[0][i][j] < 5:
                sev[(i,j)] = severity_array[0][i][j]-1
                vals = []

                for l in range(len(rgb_array)):
                    vals.append(rgb_array[l][i][j])
                rgb[(i,j)] = vals

    return sev, rgb



def declare_transformation_objs(sev_obj, tiff_objs: List):
    transformations = []
    incrs = CRS(sev_obj.crs.to_string())
    for tiff in tiff_objs:
        outcrs = CRS(tiff.crs.to_string())
        transformer = Transformer.from_crs(incrs, outcrs, always_xy=True)
        transformations.append(transformer)
    return transformations 

def close_tiffs(tiff_objs: List):
    for tiff in tiff_objs:
        tiff.close()

def normalize_tiffs(tiff_data, min_max: bool=False, mean_sdt: bool=True):
    if min_max:
        for i in range(len(tiff_data)):
            tiff = tiff_data[i]
            tiff = tiff - tiff.min()
            denominator = tiff.max()-tiff.min()
            tiff_data[i] = tiff / denominator
    if mean_sdt:
         for i in range(len(tiff_data)):
            tiff = tiff_data[i]
            mean = tiff.mean(axis=(0,1), keepdims=True)
            std = tiff.std(axis=(0,1), keepdims=True)
            tiff = tiff - mean
            tiff_data[i] = tiff / std

def make_csv(name: str, numpy_data):
    df = pd.DataFrame(numpy_data)
    df.to_csv(name + '.csv', index = False)
    


def get_numpy_from_tiffs(severity_fps: List[str], rgbs_fps: List[str], 
                            data_fps: List[str], to_normailze=True, make_csvs = True):
    """
    puts lables and pixel data from tiffs into numpy arrays
    """
    #declare stuff once
    print("opening tiffs")
    severity_data, sev_objs  = open_tiffs(severity_fps)
    rgb_data, throw_away = open_tiffs(rgbs_fps)
    tiff_data, tiff_objs = open_tiffs(data_fps)
    close_tiffs(throw_away)

    if to_normailze:
        normalize_tiffs(rgb_data)
        normalize_tiffs(tiff_data)

    lables = []
    data = []

    #get data based on current sev location and transformations
    for i in range(len(sev_objs)):
        print(f"Getting data for severity tiff {i}")
        sev_obj = sev_objs[i]
        #loop through once to get dicts of row_col values and rgb values
        sev, rgbs = get_arrays_from_dnbr6(severity_data[i], rgb_data[i])
        transformers = declare_transformation_objs(sev_obj, tiff_objs)
        for key in rgbs.keys():
            lon, lat = sev_obj.xy(key[0],key[1])
            for tiff in range(len(tiff_objs)):
                lon2, lat2 = transformers[tiff].transform(lon, lat)
                row2, col2 = tiff_objs[tiff].index(lon2, lat2)
                for k in range(len(tiff_data[tiff])):
                    rgbs[key].append(tiff_data[tiff][k][row2][col2])  

        #get surrounding data points
        number_layers = 4
        dir_vecs = gen_dir_vecs(number_layers)       
        
        
        for key in rgbs.keys():
            values = []
            row, col = key
            for r in dir_vecs:
                row_ = []
                for dir in r:
                    val = rgbs.get((dir[0]+row, dir[1]+col))
                    if val != None and values !=None:
                        row_.append(val)
                    else:
                        values = None
                if values != None:
                    values.append(row_)

            if values != None:
                data.append(values)               
                lables.append(sev[key])

    
    close_tiffs(sev_objs)
    close_tiffs(tiff_objs)    
    
    print("balacing data and calculating test cases")

    #sort into dict
    catgories = {0: [], 1: [], 2: [], 3: []}
    for i in range(len(lables)):
        
        #catgories[randint(0,3)].append(data[i])
        catgories[lables[i]].append(data[i])


    # find labble with smallest num of samples
    min_catagorie = len(catgories[0])
    print(f"Num in each catoroie {min_catagorie}")
    for key in catgories.keys():
        in_catagorie =  len(catgories[key])
        if in_catagorie < min_catagorie:
            min_catagorie = in_catagorie

    
    b_test_lables = []
    b_test_data = []

    num_test_casses = floor((min_catagorie)/5)#20% of data set is test cases

    #create balanced dataset and testset
    num_in_catagories = min_catagorie - num_test_casses
    for key in catgories.keys():
        for i in range(num_test_casses): 
            rand_index = randrange(0,num_in_catagories,1)
            b_test_data.append(catgories[key][rand_index])
            catgories[key].pop(rand_index)
        b_test_lables += [key]*num_test_casses   
    
    balanced_lables = []
    balanced_data = []
    for key in catgories.keys():
        balanced_lables += [key]*num_in_catagories
        balanced_data += catgories[key][:num_in_catagories]

    balanced_lables = numpy.array(balanced_lables)
    balanced_data = numpy.array(balanced_data)
    b_test_lables = numpy.array(b_test_lables)
    b_test_data = numpy.array(b_test_data)
    dif = "everything_4_"
    if make_csvs:
        make_csv(dif + 'balanced-severity-y_test', b_test_lables)
        shape = b_test_data.shape
        print(balanced_data.shape)
        b_test_data = b_test_data.reshape(num_test_casses*4,shape[1]*shape[2]*shape[3])#4 bc that is the number of catagories
        make_csv(dif + 'balanced-severity-x_test', b_test_data)
        make_csv(dif + 'balanced-severity-y_training', balanced_lables)
        balanced_data = balanced_data.reshape(num_in_catagories*4,shape[1]*shape[2]*shape[3])
        make_csv(dif + 'balanced-severity-x_training', balanced_data)
        return

    return balanced_lables, b_test_lables, balanced_data, b_test_data



if __name__ == "__main__":


    #geo data
    lossyear = "data\\2013\Hansen_GFC2013_lossyear_50N_130W.tif"
    treecover = "data\\2013\Hansen_GFC-2019-v1.7_treecover2000_50N_130W.tif"
    elevation = "data\\2013\\USGS_13_n43w124.tif"
    slope = "data\\2013\slope2.tif"
    stream_distance = "data\\2013\streams_zoom.tif"
    stream_degree = "data\\2013\stream_degree.tif"
    stand_age = "data\\2013\stand_age_clip.tif"
    stand_age_80 = "data\\2013\stand_age_80.tif"
    stand_age_200 = "data\\2013\stand_age_200.tif"
    aspect = "data\\2013\\aspect.tif"

    #file paths for bigger fire
    sev1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_20140706_dnbr6.tif"
    rgb1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_l8_refl.tif"

    #file paths for smaller fire
    sev2_fp = "data\\2013\or4285712358520130726\or4285712358520130726_20130703_20140706_dnbr6.tif"
    rgb2_fp = "data\\2013\or4285712358520130726\or4285712358520130726_20130703_l8_refl.tif"

    #mill fire
    sev3_fp = "data\\2017\or4293912360020170827\or4293912360020170827_20170714_20180717_dnbr6.tif"
    rgb3_fp = "data\\2017\or4293912360020170827\or4293912360020170827_20170714_l8_refl.tif"

    #big windy complex
    sev4_fp = "data\\2018\or4252812357120180715\or4252812357120180715_20180701_20190720_dnbr6.tif"
    rgb4_fp = "data\\2018\or4252812357120180715\or4252812357120180715_20180701_L8_refl.tif"

    #other fire
    sev5_fp = "data\\2013\or4261412376020130726\or4261412376020130726_20130703_20140706_dnbr6.tif"
    rgb5_fp = "data\\2013\or4261412376020130726\or4261412376020130726_20130703_l8_refl.tif"
    #y_train, y_test, x_train, x_test = get_numpy_from_tiffs([sev1_fp],[rgb1_fp], [treecover])#, slope])

    get_numpy_from_tiffs([sev1_fp, sev2_fp, sev5_fp],[rgb1_fp, rgb2_fp, rgb5_fp],[treecover, lossyear, slope, elevation, aspect, stream_distance, stand_age, stand_age_80, stand_age_200],False)
    #get_numpy_from_tiffs([sev1_fp, sev2_fp],[rgb1_fp, rgb2_fp],[treecover, lossyear, slope],False)
    #select test set more randomly






