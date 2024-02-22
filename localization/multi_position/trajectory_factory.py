from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import KDTree

class TrajectoryFactory(ABC):
    def __init__(self, use_hdf5) -> None:
        super().__init__()
        self.coordinates = None
        self.use_hdf5 = use_hdf5
    @abstractmethod
    def get_trajectory(self, query_points):
        pass
    def path_to_coordinates(self, paths):
        extension = ".hdf5" if self.use_hdf5 else ".npy"
        coordinates = []
        for path in paths:
            coordinates.append(tuple([float(x.removesuffix(extension)) for x in path.split("_")]))
        coordinates = np.stack(coordinates)
        return coordinates
    def load_data(self, path):
        assert self.coordinates is None, f"load_data can be called only once! Got self.coordinates = {self.coordinates}"
        self.coordinates = self.path_to_coordinates(path)
        self.data = {}
        for x,y,t in self.coordinates:
            self.data[(x,y)] = sorted(self.data.get((x,y), []) + [t])        

class CCWTrajectoryFactory(TrajectoryFactory):
    def __init__(self, use_hdf5, number_of_points) -> None:
        super().__init__(use_hdf5)
        self.number_of_points = number_of_points
        self.chunks = None
    def load_data(self, path):
        super().load_data(path)
        self.chunks = []
        for x,y in self.data.keys():
            maximum_index:int = len(self.data[(x,y)])//self.number_of_points*self.number_of_points
            for i in range(0,maximum_index, self.number_of_points):
                self.chunks.append([(x,y,t) for t in self.data[(x,y)][i:i+self.number_of_points]])
    def search(self, arr, x):
        low, mid, high = 0, 0, len(arr) - 1    
        while low <= high:
            mid = (high + low) // 2
            if arr[mid] < x:
                low = mid + 1
            elif arr[mid] > x:
                high = mid - 1
            else:
                return mid
        return -1
    def get_neighbors(self, coordinate):
        all_neighbors = self.data[(coordinate[0],coordinate[1])]
        index = self.search(all_neighbors, coordinate[2])
        if index == -1:
            print(f"Attention the query point has not been found! query: {coordinate} its list {self.data[(coordinate[0],coordinate[1])]}")
            breakpoint()
        #assert index != -1, f"Attention the query point has not been found! query: {coordinate} its list {self.data[(coordinate[0],coordinate[1])]}"
        return np.array(all_neighbors)[np.arange(index, index+self.number_of_points)%len(all_neighbors)]

    def get_trajectory(self, query_point):
        assert self.data is not None, f"Data is not initialized, please first call self.load_data only once to load the data"
        ccw_neighbors = self.get_neighbors(query_point)
        ccw_neighbors = np.array([(query_point[0], query_point[1], ccw_neighbor) for ccw_neighbor in ccw_neighbors])
        if ccw_neighbors.shape[0] != self.number_of_points:
            print(f"Attention: the filtered neighbors returned only {ccw_neighbors.shape[0]} instead of {self.number_of_points} requested (unfiltered got {ccw_neighbors.shape} yielding {ccw_neighbors})")
            breakpoint()
        return ccw_neighbors
    
class AllAnglesTrajectoryFactory(TrajectoryFactory):
    # it should be similar to CCWTrajectoryFactory but much less complex: it always returns all of the angles which are 64
    def __init__(self, use_hdf5) -> None:
        super().__init__(use_hdf5)
        self.chunks = None
        self.number_of_points = 64
    def load_data(self, path):
        assert self.coordinates is None, f"load_data can be called only once! Got self.coordinates = {self.coordinates}"
        self.coordinates = self.path_to_coordinates(path)
        self.data = {}
        for x,y,t in self.coordinates:
            self.data[(x,y)] = self.data.get((x,y), []) + [t]
        self.chunks = list(self.data.items())
    def get_neighbors(self, coordinate):
        all_neighbors = self.data[(coordinate[0],coordinate[1])]
        return np.array(all_neighbors)
    
    def get_trajectory(self, query_point):
        assert self.data is not None, f"Data is not initialized, please first call self.load_data only once to load the data"
        ccw_neighbors = self.get_neighbors(query_point)
        ccw_neighbors = np.array([(query_point[0], query_point[1], ccw_neighbor) for ccw_neighbor in ccw_neighbors])
        if ccw_neighbors.shape[0] != self.number_of_points:
            print(f"Attention: the filtered neighbors returned only {ccw_neighbors.shape[0]} instead of {self.number_of_points} requested (unfiltered got {ccw_neighbors.shape} yielding {ccw_neighbors})")
            breakpoint()
        return ccw_neighbors


class ConstantSpeedRandomWalkTrajectoryFactory(TrajectoryFactory):
    def __init__(self, use_hdf5:bool, number_of_steps:int, min_speed:int=1, max_speed:int=5) -> None:
        super().__init__(use_hdf5)
        self.number_of_steps = number_of_steps
        self.min_speed = min_speed
        self.max_speed = max_speed
    def load_data(self, path):
        assert self.coordinates is None, f"load_data can be called only once! Got self.coordinates = {self.coordinates}"
        self.coordinates = self.path_to_coordinates(path)
        self.tree = KDTree(self.coordinates)
        self.delta_position = np.min(np.diff(np.sort(self.coordinates,0), 0))
    def get_exact_coordinates(self, coordinates):
        _, indices = self.tree.query(coordinates)
        return self.tree.data[indices]
        
    def get_trajectory(self, query_points):
        # query_points are the initial coordinates
        speed = np.random.randint(self.min_speed,self.max_speed, size=(1,2))
        # ATTENTION: this is wrong! In this way it won't move with the constant speed that we were looking for
        displacements = np.random.choice([1,-1], size=(self.number_of_steps, 2))
        angle_displacement = np.random.rand(self.number_of_steps,1)
        displacements = speed*displacements*self.delta_position
        displacements = np.concatenate([displacements, angle_displacement], axis=0)
        coordinates = np.cumsum(np.concatenate([query_points, displacements], axis=0), axis=0)
        coordinates = self.get_exact_coordinates(coordinates)
        return coordinates