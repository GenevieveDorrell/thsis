import csv
from geoProc_latlon_cnn import *
import pandas as pd


def make_csv(name: str, data: "numpy", lables: "numpy"):
    csvfile = open(name +'.csv', 'w',newline='')
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    assert len(lables) == len(data)
    print(data[0])

    for i in range(len(data)):
        row = [lables[i]]
        for pix in data[i]:
            row = row + pix
        filewriter.writerow(row)
    csvfile.close()



if __name__ == "__main__":
    #geo data
    lossyear = "data\\2013\Hansen_GFC2013_lossyear_50N_130W.tif"
    treecover = "data\\2013\Hansen_GFC-2019-v1.7_treecover2000_50N_130W.tif"
    elevation = "data\\2013\\USGS_13_n43w124.tif"
    slope = "data\\2013\slope2.tif"

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

    y_train, y_test, x_train, x_test = get_numpy_from_tiffs([sev1_fp, sev2_fp, sev5_fp],
                                                            [rgb1_fp, rgb2_fp, rgb5_fp], 
                                                            [treecover, lossyear, slope])
    print(x_test[0])
    df = pd.DataFrame(y_test, columns = ['labes'])
    df.to_csv(r'balanced-severity-test.csv', index = False)
    #make_csv('balanced-severity-data', x_train, y_train)
    #make_csv('balanced-severity-test', x_test, y_test)