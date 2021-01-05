# import modules
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
import geopandas as gpd
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
import cartopy.crs as ccrs
import sys


class Buffer:
    def __init__(self, input_point, radius, matrix, affine_transformation):
        self.buffer = input_point.buffer(radius)
        self.inv_bool = ~rasterio.features.rasterize([(self.buffer, 1)], out_shape=elevation_shape,
                                                     transform=affine_transformation).astype(bool)
        self.matrix = matrix
        self.matrix[self.inv_bool] = np.nan
        self.maximum_elevation = []
        # Add this^ value to the printed map
        self.maximum_elevation_coords = ()

    def highest_point(self):
        row_idx, col_idx = np.unravel_index(np.nanargmax(self.matrix, ),
                                            elevation_shape)
        self.maximum_elevation.append(elevation_matrix[row_idx, col_idx])
        x_max_cord, y_max_cord = elevation_file.xy(row_idx, col_idx)
        self.maximum_elevation_coords = (x_max_cord, y_max_cord)


# # Task 1 - User input

# x_cord = int(input("please enter a numeric Easting coordinate value between 430000 - 465000: "))
# while x_cord < 430000 or x_cord > 465000:
#     x_cord = int(input(
#         "Entry not in range. Please enter a numeric easting coordinate value between 430000 - 465000: "))
# else:
#     y_cord = int(input("please enter a numeric Northing coordinate value between 80000 - 95000: "))
#     while y_cord < 80000 or y_cord > 95000:
#         y_cord = int(input(
#             "Entry not in range. Please enter a numeric Northing coordinate value between 80000 - 95000: "))
#     else:
#         print("Your location ", x_cord, " Easting and ", y_cord, " Northing is within the northing and easting "
#                                                                  "bounds ")
        
# TASK 6 - EXTENDED ENVIRONMENT

#input BNG easting and northing coordinates
x_cord = float(input("please enter a numeric Easting coordinate value: "))
y_cord = float(input("please enter a numeric Northing coordinate value: "))


# #Practice Point
# point = Point(439000, 85000) # inside polygon
#point = Point(439000, 85000) # inside polygon
#point = Point(435467.0003706848, 89877.89764861623) # basic boundary point test
#point = Point(450570.5006727509, 96489.40075877355) #boundary point test northern river. 
#point = Point(458703.8009043259, 79878.203815303283) #boundary point test
#point = Point(442533.5998012663, 92429.09739499213) # boundary point river
#point = Point(430037.8998216286, 85012.30020408472) # boundary point
#point = Point(445379.6009559168, 78917.60170240863)
#point = Point(433797.3990891299, 85464.4023485058) # another very high point is much closer

#read in the isle of wight shapefile
gdf = gpd.read_file('C:/Users/tanny/OneDrive/Documents/university year 4/Geospatial Programming/group-assignment/Material/shape/isle_of_wight.shp')

#checking correct CRS is used
gdf = gdf.to_crs(epsg=27700)

# return a linestring geoemtry of the boundary/polygon of the isle of wight
isle_of_wight_polygon = (gdf.loc[0,'geometry'])

# create point object of user input
point = Point(x_cord, y_cord)

# testing to see if inputted points are within, outside or on the border of the isle of wight
if point.touches(isle_of_wight_polygon):
    print('Your location is on the border of the isle of wight.')
elif point.within(isle_of_wight_polygon):
    print('Your location is inside the isle of wight.')
else:
    print('Your location is outside the isle of wight.')
    # terminates program
    sys.exit("your inputted coordinates are not within the isle of wight. This program will now terminate.")


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

# apply affine transformation to map pixel/cell locations to spatial positions

affine_tr = rasterio.transform.from_bounds(w_lim, s_lim,
                                           e_lim, n_lim,
                                           elevation_shape[1], elevation_shape[0])

five_km = Buffer(point, 5000, elevation_matrix, affine_tr)
five_km.highest_point()

# TASK 3 - Nearest Integrated Transport Network

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
    if five_km.buffer.contains(q_node_co):
        idx.insert(i, (node_co[0], node_co[1], node_co[0], node_co[1]), coord_id)

# Query to find  the nearest fid for the users point
for i in idx.nearest((point.x, point.y), 1):
    start_node_name = id_list[i]

# Query to find  the nearest fid for the highest point

for i in idx.nearest(five_km.maximum_elevation_coords, 1):
    dest_node_name = id_list[i]

# Printing starting and destination nodes, error handling

try:
    print('Nearest node: ', start_node_name)
    print('Destination node: ', dest_node_name)
except NameError:
    print('No nodes within buffer, please restart with different coordinates')
    exit()

# Task  4:  Shortest  Path

# Instantiate graph

g = nx.Graph()

# Add edges using road_links data to the graph

for edge in road_links:
    g.add_edge(road_links[edge]['start'], road_links[edge]['end'], fid=edge, length=road_links[edge]['length'])

# Calculate the amount of time it would take to walk each edge, irrespective of elevation, at 5km/h

for u, v in g.edges:
    g.edges[u, v]['base_time'] = (g.edges[u, v]['length']) / 1.38889

# Create a buffer slightly larger than the 5km radius, in case the dijkstra path involves briefly leaving the 5km buffer

seven_km = Buffer(point, 7000, elevation_matrix, affine_tr)

# Calculate the change in elevation for each node

for u, v in g.edges:
    u_node_coords = road_nodes[u]['coords']
    u_node_array_coords = elevation_file.index(u_node_coords[0], u_node_coords[1])
    u_node_elevation = seven_km.matrix[u_node_array_coords[0], u_node_array_coords[1]]

    v_node_coords = road_nodes[v]['coords']
    v_node_array_coords = elevation_file.index(v_node_coords[0], v_node_coords[1])
    v_node_elevation = seven_km.matrix[v_node_array_coords[0], v_node_array_coords[1]]

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

# Determine Dijkstra (shortest) path

path = nx.dijkstra_path(g, source=start_node_name, target=dest_node_name, weight='total_time')

# create a starting point (first node in ITN)

start_node_coords = road_nodes[path[0]]['coords']
start_point = Point(start_node_coords[0], start_node_coords[1])

# create a destination point (last node in index)

end_node_coords = road_nodes[path[-1]]['coords']
destination_point = Point(end_node_coords[0], end_node_coords[1])

# STEP5 - PLOTTING MAPS AND DATA

# read in raster-50k_2724246.tif file as map_tif

map_tif = rasterio.open("Material/background/raster-50k_2724246.tif")

# draw out and cut the window for background map plotting

width_of_map = 10000
height_of_map = 10000

western_map_edge, eastern_map_edge = point.x - (width_of_map / 2), point.x + (width_of_map / 2)
southern_map_edge, northern_map_edge = point.y - (height_of_map / 2), point.y + (height_of_map / 2)

# identify rows and coloumns of the pixels of the NW and SE corner in the background map
row_west, col_north, = map_tif.index(western_map_edge, northern_map_edge)
row_east, col_south = map_tif.index(eastern_map_edge, southern_map_edge)

# define map height and width
map_height = row_east - row_west + 10
map_width = col_south - col_north + 10
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

# width and height of window
elevation_height = row_east1 - row_west1 + 10
elevation_width = col_south1 - col_north1 + 10

# BNG coordinates of NW corner.
elevation_west, elevation_north = elevation_file.xy(row_west1,
                                                    col_north1,
                                                    offset='ul')
# BNG coordinates of SE corner.
elevation_east, elevation_south = elevation_file.xy(row_west1 + elevation_height,
                                                    col_north1 + elevation_width,
                                                    offset='ul')
# create a  list of the N,W,S,E extents
elevation_extents = [elevation_west,
                     elevation_east,
                     elevation_south,
                     elevation_north]

# cut the window in the elevation array
elevation_matrix_cut = elevation_file.read(1, window=Window(col_north1,
                                                            row_west1,
                                                            elevation_width,
                                                            elevation_height))

# create a GeoDataFrame of the shortest path

links = []  # this list will be used to populate the feature id (fid) column
geom = []  # this list will be used to populate the geometry column

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

# Plot background map
fig = plt.figure(figsize=(3, 3), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=crs.OSGB())

ax.imshow(background_map, origin='upper', extent=background_extents, zorder=0)
ax.set_extent(background_extents, crs=crs.OSGB())

# plot shortest path on background map
shortest_path_gpd.plot(ax=ax, edgecolor='blue', linewidth=0.5, zorder=2, label='shortest path')

# plot user input point
plt.scatter(point.x, point.y, marker="x", s=5, label='Your starting point')
# plot starting point
plt.scatter(start_point.x, start_point.y, s=5, label='The routes starting point')

# plot ending point
plt.scatter(destination_point.x, destination_point.y, s=5, label='Destination')

# plot buffer region
circle1 = plt.Circle((point.x, point.y), 5000, color='red', fill=False, linewidth=0.5, label='5km Buffer zone')
circle_buff = plt.gcf().gca().add_artist(circle1)
circle_buff.set_label('5km Buffer')

# plot elevation array
elev_img = ax.imshow(elevation_matrix_cut, origin='upper', alpha=0.3,
                     extent=elevation_extents, transform=ccrs.OSGB(), zorder=1)

# Colour bar for elevation
colour_bar = plt.colorbar(elev_img, label='Elevation')

plt.title('Isle of Wight Emergency Route Planner')

# Create North arrow
box = dict(boxstyle="rarrow,pad=0.3", fc="black", ec="k")
ax.text((point.x + 4000), (point.y + 4000), "    ", rotation=90,
        size=4,
        bbox=box)
# 5km scale - change to text fraction
plt.annotate(' ', xy=(point.x, (point.y - 4700)), xytext=((point.x + 5000), (point.y - 4700)),
             arrowprops=dict(arrowstyle="-"))
plt.text(point.x + 2000, (point.y - 4500), '5km')
plt.axhline(y=point.y - 4000, xmin=point.x, xmax=point.x + 5000)
plt.legend(loc='best', fontsize=3, bbox_to_anchor=(0.9, -0.02), ncol=3)

plt.show()
plt.savefig('route_to_highest_point.png') #save fig

# https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
# https://rasterio.readthedocs.io/en/latest/api/rasterio.transform.html