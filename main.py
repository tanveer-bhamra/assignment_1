# import modules
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
import geopandas as gpd
import matplotlib.pyplot as plt
from cartopy import crs
import rasterio
from rasterio import features
from rasterio.windows import Window
from rasterio import plot
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import networkx as nx
import numpy as np
import json
import os
from rtree import index
from scipy import sparse
import cartopy.crs as ccrs
import cartopy

# # Task 1 - User input

x_cord = int(input("please enter a numeric Easting coordinate value between 430000 - 465000: "))
while x_cord < 430000 or x_cord > 465000:
    x_cord = int(input(
        "Entry not in range. Please enter a numeric easting coordinate value between 430000 - 465000: "))
else:
    y_cord = int(input("please enter a numeric Northing coordinate value between 80000 - 95000: "))
    while y_cord < 80000 or y_cord > 95000:
        y_cord = int(input(
            "Entry not in range. Please enter a numeric Northing coordinate value between 80000 - 95000: "))
    else:
        print("Your location ", x_cord, " Easting and ", y_cord, " Northing is within the northing and easting "
                                                                  "bounds ")

point = Point(x_cord, y_cord)

# Practice Point
#point = Point(439000, 85000)

# # test point to copy and paste   435000,85000
#
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

# apply affine transformation to map pixel/cell locations to spatial positions

affine_tr = rasterio.transform.from_bounds(w_lim, s_lim,
                                           e_lim, n_lim,
                                           elevation_shape[1], elevation_shape[0])

# Converting a buffer vector geometry into raster  where cells are ones (inside 5km buffer)
# or zeros (outside 5km buffer)

buffer_raster = rasterio.features.rasterize([(buffer, 1)], out_shape=elevation_shape,
                                            transform=affine_tr)

# convert buffer matrix into boolean data type
buffer_bool = buffer_raster.astype(bool)

# invert buffer_bool matrix
buffer_bool_inv = ~buffer_bool

# clip elevation_matrix array using buffer array
clipped_elevation_matrix = elevation_matrix.copy()
clipped_elevation_matrix[buffer_bool_inv] = np.nan

# find maximum height in the buffer

row_idx, col_idx = np.unravel_index(np.nanargmax(clipped_elevation_matrix, ),
                                    elevation_shape)

# return highest point in meters in 5km buffer region

max_elevation = elevation_matrix[row_idx, col_idx]

# retrieve  the British national grid coordinates of the pixel containing the maximum elevation_matrix inside the buffer region

x_max_cord, y_max_cord = elevation_file.xy(row_idx, col_idx)

# create a point object  and list of highest point inside 5km buffer

highest_point = Point(x_max_cord, y_max_cord)
highest_point_cord = [x_max_cord, y_max_cord]

# TASK 3 - Nearest Integrated Transport Network
# test point to copy and paste   435000,85000
# highest point for test point  439007.5 85192.5 #( test points can be deleted when completed)
# highest_point_cord = [439007.5, 85192.5] #( test points can be deleted when completed)
# point = Point(435000, 85000)  #( test points can be deleted when completed)

# load itn file
itn_json = os.path.join("Material/itn/solent_itn.json")
with open(itn_json, 'r') as f:
    itn_json = json.load(f)

road_links = itn_json['roadlinks']
road_nodes = itn_json['roadnodes']

# initialize rtree
idx = index.Index()

# inserting coordinates,fid into rtree
id_list = []
for i, (coord_id, coords) in enumerate(road_nodes.items()):
    node_co = (coords['coords'])
    q_node_co = Point(node_co[0], node_co[1])
    id_list.append(coord_id)
    if buffer.contains(q_node_co):
        idx.insert(i, (node_co[0], node_co[1], node_co[0], node_co[1]), coord_id)

# Query to find  the nearest fid for the users point
for i in idx.nearest((point.x, point.y), 1):
    start_node_name = id_list[i]

# Query to find  the nearest fid for the highest point

for i in idx.nearest(highest_point_cord, 1):
    dest_node_name = id_list[i]

# Task  4:  Shortest  Path

# Determine difference in height

g = nx.Graph()

# Add edges road lines using road_links data

for edge in road_links:
    g.add_edge(road_links[edge]['start'], road_links[edge]['end'], fid=edge, length=road_links[edge]['length'])

# Calculate the amount of time it would take to walk the edge, irrespective of elevation at 5km/h

for u, v in g.edges:
    g.edges[u, v]['base_time'] = (g.edges[u, v]['length'])/1.38889

# Calculate the change in elevation along each edge
expanded_buffer = point.buffer(7500)

# Converting a buffer vector geometry into raster  where cells are ones (inside 5km buffer)
# or zeros (outside 5km buffer)

expanded_buffer_raster = rasterio.features.rasterize([(buffer, 1)], out_shape=elevation_shape,
                                            transform=affine_tr)

# convert buffer matrix into boolean data type
expanded_buffer_bool = expanded_buffer_raster.astype(bool)

# invert buffer_bool matrix
expanded_buffer_bool_inv = ~expanded_buffer_bool

expanded_buffer_search = elevation_matrix.copy()
expanded_buffer_search[buffer_bool_inv] = np.nan

for u, v in g.edges:
    u_node_coords = road_nodes[u]['coords']
    u_node_array_coords = elevation_file.index(u_node_coords[0], u_node_coords[1])
    u_node_elevation = expanded_buffer_search[u_node_array_coords[0], u_node_array_coords[1]]

    v_node_coords = road_nodes[v]['coords']
    v_node_array_coords = elevation_file.index(v_node_coords[0], v_node_coords[1])
    v_node_elevation = expanded_buffer_search[v_node_array_coords[0], v_node_array_coords[1]]

    delta_elevation = v_node_elevation - u_node_elevation
    g.edges[u, v]['delta_elevation'] = delta_elevation

# Use change in elevation to calculate the elevation-added time, add to base time to calculate the total time

for u, v in g.edges:
    if g.edges[u, v]['delta_elevation'] > 0:
        naismith_time = g.edges[u, v]['delta_elevation'] * 6
    else:
        naismith_time = 0
    total_time = g.edges[u, v]['base_time'] + naismith_time
    g.edges[u, v]['total_time'] = total_time

# print(g.edges(data=True))

# Dijkstra
#
path = nx.dijkstra_path(g, source=start_node_name, target=dest_node_name, weight='total_time')

# printing results
print('starting node: ', start_node_name)
print('destination node: ', dest_node_name)
print('Dijkstra Path: ', path)
print('Example of edge attributes:')
print(g[path[6]][path[7]])

# create a starting point (first node in ITN) 
start_node_coords = road_nodes[path[0]]['coords']
start_point = Point(start_node_coords[0], start_node_coords[1])
#create a destination point (last node in index)
end_node_coords = road_nodes[path[-1]]['coords']
destination_point = Point(end_node_coords[0], end_node_coords[1])

# STEP5 - PLOTTING MAPS AND DATA


# read in raster-50k_2724246.tif file as map_tif

map_tif = rasterio.open("Material/background/raster-50k_2724246.tif")

# draw out and cut the window for background map plotting

width_of_map = 11000
height_of_map = 11000

western_map_edge, eastern_map_edge = point.x - (width_of_map/2), point.x + (width_of_map/2)
southern_map_edge, northern_map_edge = point.y - (height_of_map/2), point.y + (height_of_map/2)

#identify rows and coloumns of the pixels of the NW and SE corner in the background map
row_west, col_north, = map_tif.index(western_map_edge, northern_map_edge)
row_east, col_south  = map_tif.index( eastern_map_edge, southern_map_edge)

#define map height and width
map_height = row_east - row_west +10
map_width = col_south - col_north +10
# print(map_width)
# print(map_height)

# create window for background map
window_map = map_tif.read(1, window=Window(col_north, row_west, 
                                           map_width, map_height))

# create and cut a window for the elevation map

# identify rows and columns of the pixels (in the eleavtion array) 
# row and column of NW corner 
row_west1, col_north1 = elevation_file.index(western_map_edge, northern_map_edge)
# row and column of SE corner
row_east1, col_south1 = elevation_file.index(eastern_map_edge, southern_map_edge)

#width and height of window
elevation_height = row_east1 - row_west1 +10
elevation_width = col_south1 - col_north1 + 10
 
# BNG coordinates of NW corner.
elevation_west, elevation_north = elevation_file.xy(row_west1,
                                                 col_north1,
                                                 offset='ul')
# BNG coordinates of SE corner.
elevation_east, elevation_south = elevation_file.xy(row_west1+elevation_height,
                                                 col_north1+elevation_width,
                                                 offset='ul')
# create a  list of the N,W,S,E extents
elevation_extents = [elevation_west,
                     elevation_east,
                     elevation_south,
                     elevation_north]

# cut the window in the elevation array
elevation_matrix_cut = elevation_file.read(1, window= Window(col_north1,
                          row_west1,
                          elevation_width,
                          elevation_height))

# create a GeoDataFrame of the shortest path

links = [] # this list will be used to populate the feature id (fid) column
geom  = [] # this list will be used to populate the geometry column

first_node = path[0]
for node in path[1:]:
    link_fid = g.edges[first_node, node]['fid']
    links.append(link_fid)
    geom.append(LineString(road_links[link_fid]['coords']))
    first_node = node

shortest_path_gpd = gpd.GeoDataFrame({'fid': links, 'geometry': geom})
shortest_path_gpd.plot()

# backgrund extents
background_extents = [western_map_edge, eastern_map_edge, southern_map_edge, northern_map_edge]

# specify map colour

palette = np.array([value for key, value in
                    map_tif.colormap(1).items()])
background_map = palette[window_map]

#plot map

# Plot background map
fig = plt.figure(figsize=(3, 3), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=crs.OSGB())

ax.imshow(background_map, origin='upper', extent=background_extents, zorder=0)
ax.set_extent(background_extents, crs=crs.OSGB())

# plot shortest path on background map
shortest_path_gpd.plot(ax=ax, edgecolor='blue', linewidth=0.5, zorder=2)

#plot user input point
plt.scatter(point.x, point.y, marker = "x", s=5)
#plot starting point
plt.scatter(start_point.x, start_point.y, s=5)

# plot ending point
plt.scatter(destination_point.x, destination_point.y, s=5)

#plot buffer region
circle1=plt.Circle((point.x,point.y),5000,color='b', fill = False, linewidth=0.5)
plt.gcf().gca().add_artist(circle1)

# plot elevation array
ax.imshow(elevation_matrix_cut, origin='upper', alpha=0.5, 
          extent=elevation_extents, transform=ccrs.OSGB(), zorder=1)


#
rasterio.plot.show(elevation_matrix)
rasterio.plot.show(clipped_elevation_matrix)
#
# # REFERENCES:

# https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
# https://rasterio.readthedocs.io/en/latest/api/rasterio.transform.html
