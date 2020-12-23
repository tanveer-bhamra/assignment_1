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
    print("Your location ", x_cord, " Easting and ", y_cord, " Northing is within the northing and easting bounds ")
        
point = Point(x_cord,y_cord)

    

# Task 2 Highest Point Identification - Identify the highest point within a 5km radius from the user location.

# read in elevation_matrix data file

elevation_file = rasterio.open("Material/elevation/SZ.asc")
elevation_matrix = elevation_file.read(1)
elevation_file_bounds = elevation_file.bounds
elevation_shape = elevation_matrix.shape

# define the bounds of the elevation_file (this info is given in elevation_file_bounds variable)
w_lim = 425000
n_lim = 100000
s_lim = 75000
e_lim = 470000


# create a 5km buffer around the inputted point object

buffer = point.buffer(5000)

#apply affine transformation to map pixel/cell locations to spatial positions

affine_tr = rasterio.transform.from_bounds(w_lim, s_lim,
                               e_lim, n_lim,
                               elevation_shape[1], elevation_shape[0])

# Converting a buffer vector geometry into raster  where cells are ones (inside 5km buffer)
# or zeros (outside 5km buffer)

buffer_raster = rasterio.features.rasterize([(buffer, 1)], out_shape = elevation_shape, 
                                                transform= affine_tr)

# convert buffer matrix into boolean data type 
buffer_bool = buffer_raster.astype(bool)

#invert buffer_bool matrix
buffer_bool_inv = ~buffer_bool

# clip elevation_matrix array using buffer array
clipped_elevation_matrix = elevation_matrix.copy()
clipped_elevation_matrix[buffer_bool_inv] = np.nan

# find maximum height in the buffer

row_idx, col_idx = np.unravel_index(np.nanargmax(clipped_elevation_matrix,), 
                                     elevation_shape)

# return highest point in meters in 5km buffer region

max_elevation = elevation_matrix[row_idx, col_idx]

# retrieve  the British national grid coordinates of the pixel containing the maximum elevation_matrix inside the buffer region

x_max_cord, y_max_cord = elevation_file.xy(row_idx, col_idx)

# create a point object  and list of highest point inside 5km buffer

highest_point = Point(x_max_cord, y_max_cord)
highest_point_cord = [x_max_cord,y_max_cord]




# TASK 3 - Nearest Integrated Transport Network


















# REFERENCES:

  # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
  # https://rasterio.readthedocs.io/en/latest/api/rasterio.transform.html
    
    












