#import rasterio
import geopandas as gpd

#from libtiff import TIFF
file_path = "DATA (2)\mtbs\\2013\or4273212351520130726\or4273212351520130726_20130703_20140706_dnbr6.tif"

tif = gpd.read_file(file_path) # open tiff file in read mode
print(tif)
# read an image in the currect TIFF directory as a numpy array
image = tif.read_image()

# read all images in a TIFF file:
for image in tif.iter_images(): 
    pass

#tif = TIFF.open('filename.tif', mode='w')
#tif.write_image(image)