import unittest
from Data_processing import *

class TestPixels(unittest.TestCase):
    def testing_row_col(self):
        """
        puts lables and pixel data from tiffs into numpy arrays
        """
        sev1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_20140706_dnbr6.tif"
        #declare stuff once
        print("opening tiff")
        severity_datas, sev_objs  = open_tiffs([sev1_fp])
        severity_data = severity_datas[0]
        sev_obj = sev_objs[0]
        row = 301
        col = 300
        lon, lat = sev_obj.xy(row,col)
        row2, col2 = sev_obj.index(lon, lat)
        print(severity_data.shape)
        self.assertTrue((row, col) == (row2, col2))

        rgb1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_l8_refl.tif"
        rgb_data, throw_away = open_tiffs([rgb1_fp])
        sev, rgbs = get_arrays_from_dnbr6(severity_data, rgb_data[0])
        self.assertEqual(severity_data[0][row2][col2]-1, sev[(row,col)])

    def testing_tif_pix(self):
        #geo data
        lossyear = "data\\2013\Hansen_GFC2013_lossyear_50N_130W.tif"
        treecover = "data\\2013\Hansen_GFC-2019-v1.7_treecover2000_50N_130W.tif"
        elevation = "data\\2013\\USGS_13_n43w124.tif"
        slope = "data\\2013\slope2.tif"
        stream_distance = "data\\2013\EucDist_hyd_2.tif"
        stream_degree = "data\\2013\stream_degree.tif"

        #file paths for bigger fire
        sev1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_20140706_dnbr6.tif"
        rgb1_fp = "data\\2013\or4273212351520130726\or4273212351520130726_20130703_l8_refl.tif"
        """
        puts lables and pixel data from tiffs into numpy arrays
        """
        #declare stuff once
        print(r"opening tiffs")
        severity_data, sev_objs  = open_tiffs([sev1_fp])
        #rgb_data, rgb_objs = open_tiffs([rgb1_fp])
        tiff_data, tiff_objs = open_tiffs([rgb1_fp, lossyear, treecover, elevation,slope, stream_distance])

        #get data based on current sev location and transformations
        for i in range(len(sev_objs)):
            print(f"Getting data for severity tiff {i}")
            sev_obj = sev_objs[i]
            #loop through once to get dicts of row_col values and rgb values
            #sev, rgbs = get_arrays_from_dnbr6(severity_data[i], rgb_data[i])
            transformers = declare_transformation_objs(sev_obj, tiff_objs)
            row = 0
            col = 0
            lon, lat = sev_obj.xy(row,col)
            vals = []
            # 0.2249, 0.323
            for tiff in range(len(tiff_objs)):
                lon2, lat2 = transformers[tiff].transform(lon, lat)
                row2, col2 = tiff_objs[tiff].index(lon2, lat2)

                for k in range(len(tiff_data[tiff])):
                    vals.append(tiff_data[tiff][k][row2][col2]) 

            close_tiffs(sev_objs)    
            close_tiffs(tiff_objs)   
 
           
            self.assertEqual([31,26,16, 112, 37, 13, 39, 0, 0, 98, 684.94867, 31.794159, 37.119854], vals)




if __name__ == "__main__":
    unittest.main()