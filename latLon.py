import rasterio as rio

infile = r"data\\2013\or4273212351520130726\or4273212351520130726_20130703_20140706_dnbr6.tif"
outfile = r'data\test_{}.tif'
coordinates = (
    (130.5, -25.5) , # lon, lat of ~centre of Australia
    (146.0, -42.0) , # lon, lat of ~centre of Tasmania
)

# Your NxN window
N = 3

# Open the raster
with rio.open(infile) as dataset:

    # Loop through your list of coords
    for i, (lon, lat) in enumerate(coordinates):

        # Get pixel coordinates from map coordinates
        py, px = dataset.index(lon, lat)
        print('Pixel Y, X coords: {}, {}'.format(py, px))

        # Build an NxN window
        window = rio.windows.Window(px - N//2, py - N//2, N, N)
        print(window)

        # Read the data in the window
        # clip is a nbands * N * N numpy array
        clip = dataset.read(window=window)

        # You can then write out a new file
        meta = dataset.meta
        meta['width'], meta['height'] = N, N
        print(meta)
        meta['transform'] = rio.windows.transform(window, dataset.transform)
        print(meta)

        #with rio.open(outfile.format(i), 'w', **meta) as dst:
            #dst.write(clip)