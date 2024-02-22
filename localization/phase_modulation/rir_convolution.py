import numpy as np
import torch

class Rir:
    '''
    Class to compute and store the RIR (Room Impulse Response) for a given absorption coefficient and wall distance.
    '''
    def __init__(self, absorption_coefficient:float, wall_distance: float, room_dimensions:float = 120.0, data_path:str = None) -> None:
        self.absorption_coefficient = absorption_coefficient
        self.wall_distance = wall_distance
        self.room_dimensions = room_dimensions
        # TODO: define the default when no data_path is given
        self.data_path = data_path
    
    def get_rir(self):
        '''
        Return the RIR for the given absorption coefficient and wall distance.
        '''
        # TODO: check if the RIR for the given absorption coefficient and wall distance is already stored in the data_path
        # If so, load it and return it
        # Otherwise compute it, save it and return it
        #
        pass



