import torch
from torch.utils.data import Dataset
import numpy as np
import os
import datetime
import pandas as pd
from EARS.io import hdf5, fast_io
import getpass

class ModulationDatasetFixedInputSound(Dataset):
    '''
    Class that assumes that the input sound is fixed and that we only need to generate the rir.
    The rir is parametrized by the absorption coefficient and the distance from the wall.
    The absorption coefficient go from 0.05 to 0.8 with a step of 0.05.
    The distance from the wall goes from 1.0 to 5.0 with a step of 0.008.
    '''
    def __init__(self, rir_data_path:str=None, step_distance:float = 0.008,
                 duration=None) -> None:
        super().__init__()

        if rir_data_path is None:
            self.rir_data_path = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "rir_bank")
        else:
            self.rir_data_path = rir_data_path
        if not os.path.exists(self.rir_data_path):
            os.makedirs(self.rir_data_path)
            
        self.absorption_coefficients = np.arange(0.05, 0.8, 0.05)
        self.distances_from_wall = np.arange(1.0, 5.0, step_distance)
        input_sound_parameters = np.load(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "input_sound_bank", "input_sound_parameters.npz"))
        if duration is not None:
            self.rec_len = int(duration*input_sound_parameters["fs"])
        else:
            self.rec_len = int(input_sound_parameters["duration"]*input_sound_parameters["fs"])
    
    def __len__(self) -> int:
        return len(self.absorption_coefficients) * len(self.distances_from_wall)
    
    def __getitem__(self, index:int) -> torch.Tensor:
        '''
        Returns the parameters necessary to compute the rir and the distance from the wall.
        '''
        absorption_coefficient_index = index % len(self.absorption_coefficients)
        distance_from_wall_index = index // len(self.absorption_coefficients)
        absorption_coefficient = self.absorption_coefficients[absorption_coefficient_index]
        distance_from_wall = self.distances_from_wall[distance_from_wall_index]
        return torch.tensor([absorption_coefficient, distance_from_wall], dtype=torch.float32), torch.tensor([distance_from_wall], dtype=torch.float32)
    
    def get_rec_len(self,):
        return self.rec_len

class ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient(ModulationDatasetFixedInputSound):
    '''
    Class that assumes that the input sound and the absorption coefficient are fixed, and that we only need to generate the rir.
    The rir is parametrized by the absorption coefficient and the distance from the wall.
    The absorption coefficient will be specified in the initializer.
    The distance from the wall goes from 1.0 to 5.0 with a step of 0.008.
    '''
    def __init__(self, absorption_coefficient:float, rir_data_path:str=None, step_distance:float = 0.008) -> None:
        super().__init__(rir_data_path=rir_data_path, step_distance=step_distance)
        self.absorption_coefficients = np.array([absorption_coefficient])
    
class ModulationDataset(Dataset):
    """
    input is a dictionary containing:
        - input_sound_parameters: a dictionary containing:
            - omega: the angular velocity 
            - coeffs: the learned amplitudes of the sources' sound
            - phi_0: the learned phases of the sources' sound
            - harmonies: the harmonies of the sources' sound
            - fs: the sampling frequency
            - duration: the duration of the sound
            - num_rotors: the number of rotors (keep it fixed at 4 !)
        - rir_parameters: a dictionary containing:
            - absorption_coefficient: a list containing the absorption coefficients of the wall
            - distance: a list containing the distance between the drone and the wall
    """
    pass

class ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2d(Dataset):
    '''
    Class that assumes that the input sound and the absorption coefficient are fixed, and that we only need to generate the rir.
    The rir is parametrized by 2D coordinates. The absorption coefficient is fixed.
    In this case we assume that all of the data is precomputed and stored in a folder.
    '''
    def __init__(self, rir_data_path:str = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "rir", "rir_indoor_4_channels_5.0_5.0_order_1_0.5_d_0.05",
                                                        "rir_indoor"), absorption_coefficient:float = 0.5,
                                                        min_coord:torch.Tensor = torch.tensor([93,93]),
                                                        max_coord:torch.Tensor = torch.tensor([407, 407]),
                                                        min_max_scaling:bool=True,
                                                        return_absorption_coefficient:bool=False,
                                                        duration=None) -> None:
        super().__init__()
        self.rir_data_path = rir_data_path
        if not os.path.exists(self.rir_data_path):
            raise ValueError(f"The path to the rir data {self.rir_data_path} does not exist.")
        # create a list of all the files in the folder
        self.single_rir_path = os.listdir(self.rir_data_path)
        input_sound_parameters = np.load(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "input_sound_bank", "input_sound_parameters.npz"))
        if duration is not None:
            self.rec_len = int(duration*input_sound_parameters["fs"])
        else:
            self.rec_len = int(input_sound_parameters["duration"]*input_sound_parameters["fs"])
        self.absorption_coefficient = absorption_coefficient
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.min_max_scaling = min_max_scaling
        self.return_absorption_coefficient = return_absorption_coefficient

    def __len__(self) -> int:
        return len(os.listdir(self.rir_data_path))
    
    def get_coordinates(self, rir_file:str) -> tuple:
        '''
        Returns the coordinates of the rir.
        '''
        coordinates = ([float(x.strip(".npy")) for x in rir_file.split("_")[-2:]])
        if not self.min_max_scaling:
            coordinates = tuple(coordinates)
        else:
            coordinates = tuple((torch.tensor(coordinates) - self.min_coord) / (self.max_coord - self.min_coord))
        # check validity of the coordinates
        if len(coordinates) != 2:
            raise ValueError("The coordinates are not valid.")
        return coordinates
    
    def __getitem__(self, index:int) -> torch.Tensor:
        '''
        Returns the parameters necessary to compute the rir and the distance from the wall.
        '''
        rir_file = self.single_rir_path[index]
        coordinates = self.get_coordinates(rir_file)
        # return torch.tensor(coordinates)
        rir = torch.from_numpy(np.load(os.path.join(self.rir_data_path, rir_file)))
        if not self.return_absorption_coefficient:
            return rir, coordinates
        return rir, coordinates, self.absorption_coefficient
    
    def get_rec_len(self,):
        return self.rec_len
    
class ModulationDatasetFixedInputSoundFixedGeometryDeformation2d(Dataset):
    '''
    Class that assumes that the input sound and the absorption coefficient are fixed, and that we only need to generate the rir.
    The rir is parametrized by 2D coordinates. The absorption coefficient is fixed.
    In this case we assume that all of the data is precomputed and stored in a folder.
    '''
    def __init__(self, rir_data_path:str = None, deformation:float = 0.5,
                                                        min_coord:torch.Tensor = torch.tensor([93,93]),
                                                        max_coord:torch.Tensor = torch.tensor([403, 403]),
                                                        min_max_scaling:bool=True,
                                                        return_deformation:bool=False,
                                                        duration=None) -> None:
        super().__init__()
        self.rir_data_path = rir_data_path
        assert rir_data_path is not None, "You need to specify the path, please set rir_data_path"
        if not os.path.exists(self.rir_data_path):
            raise ValueError(f"The path to the rir data {self.rir_data_path} does not exist.")
        # create a list of all the files in the folder
        self.single_rir_path = os.listdir(self.rir_data_path)
        input_sound_parameters = np.load(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "input_sound_bank", "input_sound_parameters.npz"))
        if duration is not None:
            self.rec_len = int(duration*input_sound_parameters["fs"])
        else:
            self.rec_len = int(input_sound_parameters["duration"]*input_sound_parameters["fs"])
        self.deformation = deformation
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.min_max_scaling = min_max_scaling
        self.return_deformation = return_deformation

    def __len__(self) -> int:
        return len(os.listdir(self.rir_data_path))
    
    def get_coordinates(self, rir_file:str) -> tuple:
        '''
        Returns the coordinates of the rir.
        '''
        coordinates = ([float(x.strip(".npy")) for x in rir_file.split("_")[-2:]])
        if not self.min_max_scaling:
            coordinates = tuple(coordinates)
        else:
            coordinates = tuple((torch.tensor(coordinates) - self.min_coord) / (self.max_coord - self.min_coord))
        # check validity of the coordinates
        if len(coordinates) != 2:
            raise ValueError("The coordinates are not valid.")
        return coordinates
    
    def __getitem__(self, index:int) -> torch.Tensor:
        '''
        Returns the parameters necessary to compute the rir and the distance from the wall.
        '''
        rir_file = self.single_rir_path[index]
        coordinates = self.get_coordinates(rir_file)
        # return torch.tensor(coordinates)
        try:
            rir = torch.from_numpy(np.load(os.path.join(self.rir_data_path, rir_file)))
        except:
            print(f"Cannot load from file {os.path.join(self.rir_data_path, rir_file)}")
            raise ValueError()
        if not self.return_deformation:
            return rir, coordinates
        return rir, coordinates, self.deformation
    
    def get_rec_len(self,):
        return self.rec_len

class ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2dOrientation(Dataset):
    '''
    Class that assumes that the input sound and the absorption coefficient are fixed, and that we only need to generate the rir.
    The rir is parametrized by 2D coordinates and orientation. The absorption coefficient is fixed.
    In this case we assume that all of the data is precomputed and stored in a folder.
    '''
    def __init__(self, rir_data_path:str = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "rir", "rir_indoor_4_channels_orientation_5.0_5.0_order_1_0.5_d_0.05_d_angle_0.39269908169872414",
                                                        "rir_indoor"), absorption_coefficient:float = 0.5,
                                                        min_coord:torch.Tensor = torch.tensor([93,93]),
                                                        max_coord:torch.Tensor = torch.tensor([403, 403]),
                                                        min_max_scaling:bool=True,
                                                        return_absorption_coefficient:bool=False,
                                                        duration=None) -> None:
        super().__init__()
        self.rir_data_path = rir_data_path
        if not os.path.exists(self.rir_data_path):
            raise ValueError(f"The path to the rir data {self.rir_data_path} does not exist.")
        # create a list of all the files in the folder
        self.single_rir_path = os.listdir(self.rir_data_path)
        input_sound_parameters = np.load(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "input_sound_bank", "input_sound_parameters.npz"))
        if duration is not None:
            self.rec_len = int(duration*input_sound_parameters["fs"])
        else:
            self.rec_len = int(input_sound_parameters["duration"]*input_sound_parameters["fs"])
        self.absorption_coefficient = absorption_coefficient
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.min_max_scaling = min_max_scaling
        self.return_absorption_coefficient = return_absorption_coefficient

    def __len__(self) -> int:
        return len(os.listdir(self.rir_data_path))
    
    def get_coordinates(self, rir_file:str) -> tuple:
        '''
        Returns the coordinates of the rir.
        '''
        coordinates = ([float(x.strip(".npy")) for x in rir_file.split("_")[-3:]])
        if not self.min_max_scaling:
            coordinates = tuple(coordinates)
        else:
            coordinates[:-1] = (torch.tensor(coordinates[:-1]) - self.min_coord) / (self.max_coord - self.min_coord)
            coordinates[-1]/=100
            coordinates = tuple(coordinates)
        # check validity of the coordinates
        if len(coordinates) != 3:
            raise ValueError("The coordinates are not valid.")
        return coordinates
    
    def __getitem__(self, index:int) -> torch.Tensor:
        '''
        Returns the parameters necessary to compute the rir and the distance from the wall.
        '''
        rir_file = self.single_rir_path[index]
        coordinates = self.get_coordinates(rir_file)
        # return torch.tensor(coordinates)
        rir = torch.from_numpy(np.load(os.path.join(self.rir_data_path, rir_file)))
        if not self.return_absorption_coefficient:
            return rir, coordinates
        return rir, coordinates, self.absorption_coefficient
    
    def get_rec_len(self,):
        return self.rec_len

def collate_fn(batch):
    '''
    Returns a batch of rirs and a batch of coordinates.
    '''
    rirs = [item[0] for item in batch]
    # pad the rirs to the same length
    max_len = max([rir.shape[-1] for rir in rirs])
    rirs = [torch.nn.functional.pad(rir, (0, max_len - rir.shape[-1])) for rir in rirs]
    coordinates = [torch.tensor(item[1]) for item in batch]
    return torch.stack(rirs), torch.stack(coordinates)

def collate_fn_absorption_coefficient(batch):
    '''
    Returns a batch of rirs and a batch of coordinates.
    '''
    rirs = [item[0] for item in batch]
    # pad the rirs to the same length
    max_len = max([rir.shape[-1] for rir in rirs])
    rirs = [torch.nn.functional.pad(rir, (0, max_len - rir.shape[-1])) for rir in rirs]
    coordinates = [torch.tensor(item[1]) for item in batch]
    absorption_coefficients = [torch.as_tensor(item[2]) for item in batch]
    return torch.stack(rirs), torch.stack(coordinates), torch.stack(absorption_coefficients)


def shear(x, angle):
    # print(f"Shearing {x} by {angle} deg")
    angle = torch.deg2rad(angle)
    c = 1/torch.tan(angle)
    R = torch.tensor([
        [1, c],
        [0, 1],
        ])
    return R@(x.to(torch.float32))

def get_filter(data_path):
    CUTOFF_DAY = datetime.datetime(2023, 12, 8, 16, 24, 3, 913227)
    def filter_files(file_path):
        return file_path.endswith('.npy') and datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(data_path,file_path))) < CUTOFF_DAY
    return filter_files

def balance_dataset_filter(data_path):
    # get list of files
    # filenames = [i for i in os.listdir(data_path) if i.endswith('.npy')]
    filenames = [i for i in fast_io.get_listed_files(data_path) if i.endswith('.npy')]
    
    # convert filenames to coordinates
    coordinates = [tuple(map(float, i.removesuffix('.npy').split('_'))) for i in filenames]

    # convert coordinates to a pandas dataframe
    df = pd.DataFrame(coordinates, columns=['x', 'y', 't'])

    counts = df.groupby(['x', 'y'])['t'].transform('nunique')

    final_df = df[counts == 128]

    # create a set of unique coordinates from the final dataframe
    coordinate_set = set(map(tuple, final_df[['x', 'y', 't']].to_numpy()))

    # return function that returns True if keep datapoint, False otherwise 
    def filter_files(file_path):
        if not file_path.endswith('.npy'):
            return False
        coordinate = tuple(map(float, file_path.removesuffix('.npy').split('_')))
        return coordinate in coordinate_set

    return filter_files

class Localization2dGivenOrientationDataset(Dataset):
    def __init__(self, 
                 data_path:str, absorption_coefficient:float = 0.5,
                 min_coord:torch.Tensor = torch.tensor([0.93,0.93]),
                 max_coord:torch.Tensor = torch.tensor([4.07, 4.07]),
                 min_max_scaling:bool=True,
                 return_absorption_coefficient:bool=False,
                 duration=None,
                 return_deformation:bool = False,
                 deformation = None, shear_angle = None, 
                 use_newton_cluster:bool = False,
                 skip_bad_files:bool=False) -> None:
        assert deformation is not None or not return_deformation, "You need to specify the deformation"
        super().__init__()
        self.data_path = data_path
        self.skip_bad_files = skip_bad_files
        if not os.path.exists(self.data_path):
            raise ValueError(f"The path to the data {self.data_path} does not exist.")
        
        self.use_newton_cluster = use_newton_cluster

        self.single_data_path = list(filter(lambda x: x.endswith(".hdf5"), fast_io.get_listed_files(data_path)))
        

        input_sound_parameters = np.load(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "input_sound_bank", "input_sound_parameters.npz"))
        if duration is not None:
            self.rec_len = int(duration*input_sound_parameters["fs"])
        else:
            self.rec_len = int(input_sound_parameters["duration"]*input_sound_parameters["fs"])
        self.absorption_coefficient = absorption_coefficient
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.min_max_scaling = min_max_scaling
        self.return_absorption_coefficient = return_absorption_coefficient
        self.return_deformation = return_deformation
        self.deformation = deformation
        self.shear_angle = None if shear_angle is None else torch.tensor(shear_angle)

    def __len__(self) -> int:
        return len(self.single_data_path)
    
    def get_coordinates(self, data_file:str) -> tuple:
        coordinates = ([float(x.removesuffix(self.extension))/10**8 for x in data_file.split("_")[:-1]])
        orientation = float(data_file.removesuffix(self.extension).split("_")[-1])/10**8

        if self.shear_angle is not None:
            coordinates = tuple(shear(torch.tensor(coordinates).unsqueeze(-1), -self.shear_angle))
        
        if not self.min_max_scaling:
            coordinates = tuple(coordinates)
        else:
            coordinates = tuple((torch.tensor(coordinates) - self.min_coord) / (self.max_coord - self.min_coord))
        # check validity of the coordinates
        if len(coordinates) != 2:
            raise ValueError(f"The coordinates are not valid. Got coordinates: {coordinates}")
        assert (all(x>=0 or abs(x)<1e-2 for x in coordinates)) \
                and (all(x<=1 or (x-1)<=1e-2 for x in coordinates)),\
            f"in file {data_file} got coordinates {coordinates} in data_path {self.data_path} min {self.min_coord} max {self.max_coord}"
        return coordinates, orientation
    
    def __getitem__(self, index:int) -> torch.Tensor:
        data_file = self.single_data_path[index]
        coordinates, orientation = self.get_coordinates(data_file)
        # return torch.tensor(coordinates)
        if self.use_newton_cluster:
            root_path = self.data_path if index<self.separator else self.data_path.replace("gabrieles", "tamir.shor", 1)
        else:
            root_path = self.data_path
        try:
            if self.extension == '.hdf5':
                data = hdf5.load_torch(os.path.join(root_path, data_file))
            elif self.extension == '.npy':
                data = torch.from_numpy(np.load(os.path.join(root_path, data_file)))
            else:
                raise ValueError(f"Extension not recognized. Got {self.extension}. Choose among .hdf5 and .npy")
            if torch.any(torch.isnan(data)):
                print(f"{os.path.join(self.data_path, data_file)} contains nan values!")
                raise ValueError(f"{os.path.join(self.data_path, data_file)} has NaN values")
            if not self.return_absorption_coefficient and not self.return_deformation:
                return data, coordinates, orientation
            elif self.return_absorption_coefficient and not self.return_deformation:
                return data, coordinates, self.absorption_coefficient, orientation
            elif not self.return_absorption_coefficient and self.return_deformation:
                return data, coordinates, self.deformation, orientation
            else:
                return data, coordinates, self.absorption_coefficient, self.deformation, orientation
        except:
            print(f"Cannot read {os.path.join(self.data_path, data_file)}")
            if not self.skip_bad_files:
                raise ValueError()
        
    
    def get_rec_len(self):
        return self.rec_len

def collate_fn_orientation(batch):
    '''
    Returns a batch of rirs and a batch of coordinates.
    '''
    rirs = [item[0] for item in batch]
    # pad the rirs to the same length
    max_len = max([rir.shape[-1] for rir in rirs])
    rirs = [torch.nn.functional.pad(rir, (0, max_len - rir.shape[-1])) for rir in rirs]
    coordinates = [torch.tensor(item[1]) for item in batch]
    orientations = [torch.as_tensor(item[2], dtype=torch.float64) for item in batch]
    return torch.stack(rirs), torch.stack(coordinates), torch.stack(orientations)

def collate_fn_absorption_coefficient_orientation(batch):
    '''
    Returns a batch of rirs and a batch of coordinates.
    '''
    rirs = [item[0] for item in batch]
    # pad the rirs to the same length
    max_len = max([rir.shape[-1] for rir in rirs])
    rirs = [torch.nn.functional.pad(rir, (0, max_len - rir.shape[-1])) for rir in rirs]
    coordinates = [torch.tensor(item[1]) for item in batch]
    absorption_coefficients = [torch.as_tensor(item[2]) for item in batch]
    orientations = [torch.as_tensor(item[3], dtype=torch.float64) for item in batch]
    return torch.stack(rirs), torch.stack(coordinates), torch.stack(absorption_coefficients), torch.stack(orientations)

def collate_fn_deformation_orientation(batch):
    '''
    Returns a batch of rirs and a batch of coordinates.
    '''
    if batch is None:
        return None, None, None, None
    good_batch_elements = list(filter(lambda x: x is not None, batch))
    if len(good_batch_elements) == 0:
        return None, None, None, None
    rirs = [item[0] for item in good_batch_elements]
    # pad the rirs to the same length
    max_len = max([rir.shape[-1] for rir in rirs])
    rirs = [torch.nn.functional.pad(rir, (0, max_len - rir.shape[-1])) for rir in rirs]
    coordinates = [torch.tensor(item[1]) for item in good_batch_elements]
    deformations = [torch.as_tensor(item[2]) for item in good_batch_elements]
    orientations = [torch.as_tensor(item[3], dtype=torch.float64) for item in good_batch_elements]
    return torch.stack(rirs), torch.stack(coordinates), torch.stack(deformations), torch.stack(orientations)