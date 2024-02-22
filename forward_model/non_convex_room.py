from shapely import buffer
from shapely.geometry import Polygon, Point
import numpy as np

def get_bounding_box(corners_list):
    min_x = np.min(corners_list[:, 0])
    max_x = np.max(corners_list[:, 0])
    min_y = np.min(corners_list[:, 1])
    max_y = np.max(corners_list[:, 1])
    return min_x, max_x, min_y, max_y

def get_grid(min_x: float, max_x: float, min_y: float, max_y: float, grid_size: int, polygon: Polygon,
             offset: float = 0.):
    x = np.linspace(min_x+offset, max_x-offset, grid_size)
    y = np.linspace(min_y+offset, max_y-offset, grid_size)
    xy = np.meshgrid(x, y)
    xy = np.stack(xy, axis=-1)
    xy = xy.reshape(-1, 2)
    xy = np.array([point for point in xy if Point(point).within(polygon)])
    return xy

def get_grid_points(polygon, grid_size, offset=0.):
    polygon_coordinates = np.array(polygon.exterior.coords)
    min_x, max_x, min_y, max_y = get_bounding_box(polygon_coordinates)
    xy = get_grid(min_x, max_x, min_y, max_y, grid_size, polygon, offset)
    return xy

def get_points_in_polygon_from_corners(corners_list:list, number_of_points_per_side:int, distance_from_wall:float,
                                       offset:float = 0.0):
    polygon = Polygon(corners_list)
    buffered_polygon = buffer(polygon, -distance_from_wall)
    xy = get_grid_points(buffered_polygon, number_of_points_per_side, offset=offset)
    return xy