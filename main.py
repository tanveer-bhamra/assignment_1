from shapely.geometry import Polygon
from shapely.geometry import Point

# Task 1 User Input


# can add error handling using while loop , and gate, try and except, with a recursion
x_point = float(input("Enter an Easting coordinate value: "))
y_point = float(input("Enter a Northing coordinate : "))

user_loc = Point(x_point, y_point)
island_mbr = Polygon([(430000, 95000), (430000, 80000), (465000, 80000), (465000, 95000)])

if user_loc.within(island_mbr):
    pass
else:
    print('Not within the softwares boundary ')
    # recursion to add to the function
