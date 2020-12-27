# import modules
from shapely.geometry import Polygon
from shapely.geometry import Point
import rasterio
from rasterio import features
from rasterio.windows import Window
from rasterio import plot
import numpy as np
import json
import os
from rtree import index

# # Task 1 - User input
#
# x_cord = int(input("please enter a numeric easting coordinate value between 430000 - 465000: "))
# y_cord = int(input("please enter a numeric Northing coordinate value between 80000 - 95000: "))
#
# while y_cord < 80000 or y_cord > 95000 or x_cord < 430000 or x_cord > 465000:
#     x_cord = int(input("Insufficent Easting coordinate. Please enter a numeric easting coordinate value between 430000 - 465000: "))
#     y_cord = int(input("Insufficent Northing coordinate.Please enter a numeric Northing coordinate value between 80000 - 95000: "))
# else:
#     print("Your location ", x_cord, " Easting and ", y_cord, " Northing is within the northing and easting bounds ")
#
# point = Point(x_cord,y_cord)
#
# # test point to copy and paste   435000,85000
#
# # Task 2 Highest Point Identification - Identify the highest point within a 5km radius from the user location.
#
# # read in elevation_matrix data file
#
# elevation_file = rasterio.open("Material/elevation/SZ.asc")
# elevation_matrix = elevation_file.read(1)
# elevation_file_bounds = elevation_file.bounds
# elevation_shape = elevation_matrix.shape
#
# # define the bounds of the elevation_file (this info is given in elevation_file_bounds variable)
# w_lim = 425000
# n_lim = 100000
# s_lim = 75000
# e_lim = 470000
#
#
# # create a 5km buffer around the inputted point object
#
# buffer = point.buffer(5000)
#
# #apply affine transformation to map pixel/cell locations to spatial positions
#
# affine_tr = rasterio.transform.from_bounds(w_lim, s_lim,
#                                e_lim, n_lim,
#                                elevation_shape[1], elevation_shape[0])
#
# # Converting a buffer vector geometry into raster  where cells are ones (inside 5km buffer)
# # or zeros (outside 5km buffer)
#
# buffer_raster = rasterio.features.rasterize([(buffer, 1)], out_shape = elevation_shape,
#                                                 transform= affine_tr)
#
# # convert buffer matrix into boolean data type
# buffer_bool = buffer_raster.astype(bool)
#
# #invert buffer_bool matrix
# buffer_bool_inv = ~buffer_bool
#
# # clip elevation_matrix array using buffer array
# clipped_elevation_matrix = elevation_matrix.copy()
# clipped_elevation_matrix[buffer_bool_inv] = np.nan
#
# # find maximum height in the buffer
#
# row_idx, col_idx = np.unravel_index(np.nanargmax(clipped_elevation_matrix,),
#                                      elevation_shape)
#
# # return highest point in meters in 5km buffer region
#
# max_elevation = elevation_matrix[row_idx, col_idx]
#
# # retrieve  the British national grid coordinates of the pixel containing the maximum elevation_matrix inside the buffer region
#
# x_max_cord, y_max_cord = elevation_file.xy(row_idx, col_idx)
#
# # create a point object  and list of highest point inside 5km buffer
#
# highest_point = Point(x_max_cord, y_max_cord)
# highest_point_cord = [x_max_cord,y_max_cord]
#


# TASK 3 - Nearest Integrated Transport Network
# test point to copy and paste   435000,85000
# highest point for test point  439007.5 85192.5 #( test points can be deleted when completed)
highest_point_cord = [439007.5, 85192.5] #( test points can be deleted when completed)
point = Point(435000, 85000)  #( test points can be deleted when completed)

# load itn file
itn_json = os.path.join("Material/itn/solent_itn.json")
with open(itn_json, 'r') as f:
    itn_json = json.load(f)

road_nodes = itn_json['roadnodes']
node_coord = []

# initialize rtree
idx = index.Index()

# inserting coordinates,fid into rtree
id_list = []
for i, (coord_id, coords) in enumerate(road_nodes.items()):
    node_co = (coords['coords'])
    id_list.append(coord_id)
    idx.insert(i, (node_co[0], node_co[1], node_co[0], node_co[1]), coord_id)


# Query to find  the nearest fid for the highest point
# fid for highest test point osgb4000000026227672
for i in idx.nearest((highest_point_cord), 1):
    highest_node = id_list[i]
    print(highest_node)

# Query to find  the nearest fid for the users point
# fid for  user test point osgb4000000026141631
for i in idx.nearest((point.x,point.y), 1):
    user_node = id_list[i]
    print(user_node)

#Task  4:  Shortest  Path

#load Json road links
#load json roadnodes
#create network graph

# REFERENCES:

# https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
# https://rasterio.readthedocs.io/en/latest/api/rasterio.transform.html
