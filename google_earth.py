from gpx_parser import get_latlon
import ee 
#sentilan
#landsat
#looking onto high resolution satalites
#INEGI
#INEGI
#https://www.inegi.org.mx/temas/topografia/#Descargas
#sate: Oaxaca
#County (municipio) : San Felipe Usila
latlon = get_latlon("data/Waypoints_23-AUG-19.gpx")
print(latlon)
