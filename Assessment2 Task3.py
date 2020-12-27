# import modules

from shapely.geometry import Polygon
from shapely.geometry import Point
import rasterio
from rasterio import features
from rasterio.windows import Window
import numpy as np
from rtree import index
import json


if __name__ == '__main__':

    # Task 1 - User input

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

    # retrieve  the British national grid coordinates of the pixel containing the maximum elevation_matrix inside the
    # buffer region

    x_max_cord, y_max_cord = elevation_file.xy(row_idx, col_idx)

    # create a point object  and list of highest point inside 5km buffer

    highest_point = Point(x_max_cord, y_max_cord)
    highest_point_cord = [x_max_cord, y_max_cord]

    # TASK 3 - Nearest Integrated Transport Network

    # Read the ITN files

    solent_itn = json.load(open('material/itn/solent_itn.json'))
    roads = solent_itn['road']
    road_links = solent_itn['roadlinks']
    road_nodes = solent_itn['roadnodes']
    route_info = solent_itn['routeinfo']

    # Finding Nearest nodes

    idx = index.Index()  # Create index
    for i, (node_id, coords) in enumerate(road_nodes.items()):  # Iterate through road_nodes, enumerating for indexing
        (x, y) = coords.get('coords')  # Extract coordinates
        node_as_point = Point(x, y)  # Format as point
        if buffer.contains(node_as_point):  # Test if each node is within 5km buffer
            idx.insert(i, (x, y), node_id)  # Add node to spatial index

    q_point = (point.x, point.y)  # Format query for starting point
    q_dest = (x_max_cord, y_max_cord)  # Format query for destination

    try:
        node_nearest_point_id = list(idx.nearest(q_point, num_results=1, objects=True))[0].object  # Retrieve near id
        node_nearest_point_coords = road_nodes[node_nearest_point_id]['coords']  # Find matching coordinates
    except IndexError:
        print('No transportation nodes detected in area')  # Error handling for exceptions

    node_nearest_dest_id = list(idx.nearest(q_dest, num_results=1, objects=True))[0].object  # Retrieve destination id
    node_nearest_dest_coords = road_nodes[node_nearest_dest_id]['coords']  # Find matching coordinates

    print(node_nearest_point_id, node_nearest_point_coords)
    print(node_nearest_dest_id, node_nearest_dest_coords)





    # nearest_target_point_fid = list(idx.nearest(highest_point, num_results=1, objects=True))[0].object

        # print(node_id, coords.get('coords'))
        # print(node_id, list(coords.values())[0])

        # node_as_point = Point(x, y)
        # if not buffer.contains(node_as_point):
        #     continue
        # idx.insert(i, x, y)
    #
    # print(idx)

    # try:
    #     nearest_origin_point_fid = list(idx.nearest(point, num_results=1,
    #                                                 objects=True))[0].object
    # except IndexError as e:
    #     clear_output()
    #     display(HTML("<h1><center> It seems that there are no transportation nodes\
    #                   near you </center></h1>"))
    #     display(HTML("<h1><center> Please, relocate and refresh the page \
    #                   </center></h1>"))
    #     start_point = ("<h2><center>" + "The point you entered is at " +
    #                    str(int(point[0])) + ", " + str(int(point[1])) +
    #                    "</center></h2>")
    #     display(HTML(start_point))
    #     while True:
    #         pass
    #
    # nearest_target_point_fid = list(idx.nearest(highest_point,
    #                                             num_results=1, objects=True))[0].object
    # nearest_target_coords = road_nodes[nearest_target_point_fid]['coords']
    # nearest_origin_coords = road_nodes[nearest_origin_point_fid]['coords']



    # REFERENCES:

    # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
    # https://rasterio.readthedocs.io/en/latest/api/rasterio.transform.html

