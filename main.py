# import modules

from shapely.geometry import Polygon
from shapely.geometry import Point
import rasterio
from rasterio import features
from rasterio.windows import Window
import numpy as np



# Task 1 User Input

def user_input():
    while True:
        try:
            x_point = float(input("Enter an Easting coordinate value: "))
            y_point = float(input("Enter a Northing coordinate : "))
            user_loc = Point(x_point, y_point)
            island_mbr: Polygon = Polygon([(430000, 95000), (430000, 80000), (465000, 80000), (465000, 95000)])
            break
        except:
            print('Please enter a valid Easting and Northing ')
            user_input()
            
    if user_loc.within(island_mbr):
        print('inside')
    else:
        print("Not within the software's boundary\nBoundary extends: \n 430000 - 465000 East\n 80000 - 95000 North"
              " \n Please input a new coordinates\n\n ")
        user_input()
        # quit software
    return user_loc, print('within boundary')  # remove the print when program works


point = user_input()

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

