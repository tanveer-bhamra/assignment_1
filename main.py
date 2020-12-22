# import modules

from shapely.geometry import Polygon
from shapely.geometry import Point
import rasterio
from rasterio import features
from rasterio.windows import Window
import numpy as np

# Task 1 - User input

x_cord = int(input("please enter a numeric easting coordinate value between 430000 - 465000: "))
y_cord = int(input("please enter a numeric Northing coordinate value between 80000 - 95000: "))

while y_cord < 80000 or y_cord > 95000 or x_cord < 430000 or x_cord > 465000:
    x_cord = int(input("Insufficent Easting coordinate. Please enter a numeric easting coordinate value between 430000 - 465000: "))
    y_cord = int(input("Insufficent Northing coordinate.Please enter a numeric Northing coordinate value between 80000 - 95000: "))
else:
    print("Your location ", x_cord, " Easting and ", y_cord, " Northing is within the northing and easting limits ")
        
point = Point(x_cord,y_cord)

    

# Task 2 Highest Point Identification - Identify the highest point within a 5km radius from the user location.

# read in elevation data file

elevation_file = rasterio.open("Material/elevation/SZ.asc")
elevation = elevation_file.read(1)
elevation_file_bounds = elevation_file.bounds

# define the bounds of the elevation file (this info is given in elevation_file_bounds variable)
w_lim = 425000
n_lim = 100000
s_lim = 75000
e_lim = 470000

# CREATE A WINDOW

# define row and cloumn offset
row_off, col_off  = elevation_file.index(w_lim, n_lim)

# define the width and height of the window
height = elevation_file.height  #rows
width = elevation_file.width #coloumns

elevation_window = elevation_file.read(1, window=Window(col_off, row_off, width, height))

# print(elevation_window.shape)


# create a 5km buffer around the inputted point object

buffer = point.buffer(5000)












