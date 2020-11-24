# Parser for gpx files

# Library for reading gpx files and parsing latlon data
import gpxpy
import gpxpy.gpx

def get_latlon(input_f):
    try:
        gpx_f = open(input_f, 'r')
    except FileNotFoundError:
        print("Could not find file " + input_f + " - exiting...")
        exit(1)
    
    gpx = gpxpy.parse(gpx_f)
    print(gpx)
    lat_long = []
    # Parse latlon data and store it as a list of tuples
    for tracks in gpx.tracks:
        for segment in tracks.segments:
            for point in segment.points:
                lat_long.append((point.latitude, point.longitude))
                if time_1 != None and point.time != None:
                    tdeltas.append((point.time - time_1).total_seconds())

    # Reduce resolution of data to increase processing speed
    lat_long = lat_long[0::10]
    return lat_long
