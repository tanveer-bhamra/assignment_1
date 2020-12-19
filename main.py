from shapely.geometry import Polygon
from shapely.geometry import Point


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
# Task 2
