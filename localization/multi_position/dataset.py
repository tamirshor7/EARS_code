from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import numpy as np
from EARS.io import hdf5, fast_io
from EARS.localization.multi_position import trajectory_factory
import getpass
import concurrent.futures

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

def shear(x, angle):
    # print(f"Shearing {x} by {angle} deg")
    angle = torch.deg2rad(angle)
    c = 1/torch.tan(angle)
    R = torch.tensor([
        [1, c],
        [0, 1],
        ])
    return R@(x.to(torch.float32))

class MultiPositionDataset(Dataset):
    def __init__(self, 
                 data_path:str, tf: trajectory_factory.TrajectoryFactory,
                 absorption_coefficient:float = 0.5,
                 min_coord:torch.Tensor = torch.tensor([0.93,0.93]),
                 max_coord:torch.Tensor = torch.tensor([4.07, 4.07]),
                 min_max_scaling:bool=True,
                 return_absorption_coefficient:bool=False,
                 duration=None,
                 return_deformation:bool = False,
                 deformation = None, shear_angle = None, filter_angles:bool=False,
                 use_newton_cluster:bool = False,
                 paths_list=None) -> None:
        assert deformation is not None or not return_deformation, "You need to specify the deformation"
        super().__init__()
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            raise ValueError(f"The path to the data {self.data_path} does not exist.")
        
        self.use_newton_cluster = use_newton_cluster
        if paths_list is not None:
            self.extension = ".hdf5" if paths_list[0].endswith(".hdf5") else ".npy"
            self.single_data_path = paths_list
            if self.use_newton_cluster:
                user_name = getpass.getuser()
                summary_file_path_root = data_path.replace("gabrieles", user_name)
                summary_file_path_first_partition = os.path.join(summary_file_path_root, "first_partition")
                # self.first_partition = set(filter(lambda x: x.endswith(".hdf5"), os.listdir(data_path)))
                self.first_partition = set(filter(lambda x: x.endswith(".hdf5"), fast_io.get_listed_files(data_path, summary_file_path=summary_file_path_first_partition)))
                summary_file_path_second_partition = os.path.join(summary_file_path_root, "second_partition")
                # second_partition = set(filter(lambda x: x.endswith(".hdf5"), os.listdir(data_path.replace("gabrieles", "tamir.shor",1))))
                second_partition = set(filter(lambda x: x.endswith(".hdf5"), fast_io.get_listed_files(data_path.replace("gabrieles", "tamir.shor",1), summary_file_path=summary_file_path_second_partition)))
        elif self.use_newton_cluster:
            self.extension = ".hdf5"
            user_name = getpass.getuser()
            summary_file_path_root = data_path.replace("gabrieles", user_name)
            summary_file_path_first_partition = os.path.join(summary_file_path_root, "first_partition")
            # self.first_partition = set(filter(lambda x: x.endswith(".hdf5"), os.listdir(data_path)))
            self.first_partition = set(filter(lambda x: x.endswith(".hdf5"), fast_io.get_listed_files(data_path, summary_file_path=summary_file_path_first_partition)))
            summary_file_path_second_partition = os.path.join(summary_file_path_root, "second_partition")
            # second_partition = set(filter(lambda x: x.endswith(".hdf5"), os.listdir(data_path.replace("gabrieles", "tamir.shor",1))))
            second_partition = set(filter(lambda x: x.endswith(".hdf5"), fast_io.get_listed_files(data_path.replace("gabrieles", "tamir.shor",1), summary_file_path=summary_file_path_second_partition)))
            self.single_data_path = list(self.first_partition.union(second_partition))
        elif not filter_angles: 
            # create a list of all the files in the folder
            hdf5_datasets = set(["/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/orig_64_angles/default_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/non_convex_room/non_convex_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/small_dense_room/default_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/shifted_room/default_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/non_convex_room/shifted_non_convex_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/asymmetric_non_convex_room/non_convex_5.0_5.0_order_1_0.5_d_0.05"
                                 ])
            self.extension = ".hdf5" if self.data_path in hdf5_datasets else ".npy"
            # self.single_data_path = list(filter(lambda x: x.endswith(self.extension), os.listdir(self.data_path)))
            self.single_data_path = list(filter(lambda x: x.endswith(self.extension), fast_io.get_listed_files(self.data_path)))
        else:
            angle_filter = balance_dataset_filter(self.data_path)
            # self.single_data_path = list(filter(angle_filter, os.listdir(self.data_path)))
            self.single_data_path = list(filter(angle_filter, fast_io.get_listed_files(self.data_path)))

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

        assert isinstance(tf, (trajectory_factory.CCWTrajectoryFactory, trajectory_factory.AllAnglesTrajectoryFactory)), f"Currently the only supported trajectory factory class is CCWTrajectoryFactory, but got {type(tf)} class"
        self.trajectory_factory = tf
        self.trajectory_factory.load_data(self.single_data_path)

        self.chunks = self.trajectory_factory.chunks

    def __len__(self) -> int:
        #return len(self.single_data_path)
        return len(self.chunks)
    
    def get_coordinates(self, data_file:str) -> tuple:
        # print(f"processing {data_file}")
        coordinates = tuple([float(x.removesuffix(self.extension)) for x in data_file.split("_")[:-1]])
        orientation = float(data_file.removesuffix(self.extension).split("_")[-1])
        if len(coordinates) != 2:
            raise ValueError(f"The coordinates are not valid. Got coordinates: {coordinates}")

        return coordinates, orientation

    def process_coordinates(self, coordinates):
        coordinates = tuple([float(x)/10**8 for x in coordinates])
        if self.shear_angle is not None:
            coordinates = tuple(shear(torch.tensor(coordinates).unsqueeze(-1), -self.shear_angle))
        
        if not self.min_max_scaling:
            coordinates = tuple(coordinates)
        else:
            coordinates = tuple((torch.tensor(coordinates) - self.min_coord) / (self.max_coord - self.min_coord))
        assert (all(x>=0 for x in coordinates) or all(abs(x)<1e-4 for x in coordinates)) \
                and (all(x<=1 for x in coordinates) or all((x-1)<=1e-4 for x in coordinates)),\
            f"got coordinates {coordinates} in data_path {self.data_path} min {self.min_coord} max {self.max_coord}"
        return coordinates

    def chunk_to_path(self, chunk):
        # spatial_coordinates is a tuple
        spatial_coordinates = chunk[0]
        to_path = lambda angle: f"{int(spatial_coordinates[0])}_{int(spatial_coordinates[1])}_{int(angle)}{self.extension}"
        paths = [to_path(angle) for angle in chunk[1]]
        return paths
    
    def chunk_to_coordinates(self, chunk):
        spatial_coordinates = chunk[0]
        to_coordinate = lambda angle: (spatial_coordinates[0], spatial_coordinates[1], angle)
        coordinates = [to_coordinate(angle) for angle in chunk[1]]
        return coordinates
    
    def load_data(self, path):
        if self.use_newton_cluster:
            root_path = self.data_path if path in self.first_partition else self.data_path.replace("gabrieles", "tamir.shor", 1)
        else:
            root_path = self.data_path

        try:
            if self.extension == '.hdf5':
                d = hdf5.load_torch(os.path.join(root_path, path))
            elif self.extension == '.npy':
                d = torch.from_numpy(np.load(os.path.join(root_path, path)))
            else:
                raise ValueError(f"Extension not recognized. Got {self.extension}. Choose among .hdf5 and .npy")
        except:
            print(f"Cannot read {os.path.join(self.data_path, path)}")
            raise ValueError()
        return d
    
    def load_data_parallel(self, all_paths):
        data = []
        # print(f"[debug] About to load data with multiple threads...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.load_data, path) for path in all_paths]
            # counter = 0
            for future in concurrent.futures.as_completed(futures):
                # print(f"[debug] {counter} done")
                # counter += 1
                try:
                    result = future.result()
                    data.append(result)
                except Exception as e:
                    # Handle exceptions if needed
                    print(f"Got exception {e}")
                    raise e
        return data
    
    def __getitem__(self, index:int) -> torch.Tensor:
        # data_file = self.single_data_path[index]
        # coordinates_to_estimate, orientation = self.get_coordinates(data_file)
        # coordinates_with_orientation = np.array((coordinates_to_estimate[0], coordinates_to_estimate[1], orientation))
        # all_coordinates = self.trajectory_factory.get_trajectory(coordinates_with_orientation)
        # coordinates, orientation = all_coordinates[...,:-1], all_coordinates[...,-1]
        # coordinates_to_estimate = [self.process_coordinates(coordinate) for coordinate in coordinates]
        # coordinates_to_estimate = coordinates_to_estimate[0]
        # orientation = orientation.astype(np.float64)/10**8
        # all_paths = [f"{int(coord[0])}_{int(coord[1])}_{int(coord[2])}{self.extension}" for coord in all_coordinates]
        chunk = self.chunks[index]
        #breakpoint()
        #all_paths = list(map(self.coordinates_to_path, all_coordinates))
        all_paths = self.chunk_to_path(chunk)
        
        all_coordinates = self.chunk_to_coordinates(chunk)
        all_coordinates = np.stack(all_coordinates)
        coordinates_to_estimate = self.process_coordinates(all_coordinates[0,:2])
        orientation = all_coordinates[...,-1].astype(np.float64)/10**8

        # data = []
        # for path in all_paths:
        #     if self.use_newton_cluster:
        #         root_path = self.data_path if path in self.first_partition else self.data_path.replace("gabrieles", "tamir.shor", 1)
        #     else:
        #         root_path = self.data_path

        #     try:
        #         if self.extension == '.hdf5':
        #             d = hdf5.load_torch(os.path.join(root_path, path))
        #         elif self.extension == '.npy':
        #             d = torch.from_numpy(np.load(os.path.join(root_path, path)))
        #         else:
        #             raise ValueError(f"Extension not recognized. Got {self.extension}. Choose among .hdf5 and .npy")
        #     except:
        #         print(f"Cannot read {os.path.join(self.data_path, path)}")
        #         raise ValueError()
        #     data.append(d)
        data = self.load_data_parallel(all_paths)
        max_len = max(d.shape[-1] for d in data)
        data = torch.stack([torch.nn.functional.pad(d, (0,max_len-d.shape[-1])) for d in data])
                
        if not self.return_absorption_coefficient and not self.return_deformation:
            return data, coordinates_to_estimate, orientation
        elif self.return_absorption_coefficient and not self.return_deformation:
            return data, coordinates_to_estimate, self.absorption_coefficient, orientation
        elif not self.return_absorption_coefficient and self.return_deformation:
            return data, coordinates_to_estimate, self.deformation, orientation
        else:
            return data, coordinates_to_estimate, self.absorption_coefficient, self.deformation, orientation
    
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
    rirs = [item[0] for item in batch]
    # pad the rirs to the same length
    max_len = max([rir.shape[-1] for rir in rirs])
    rirs = [torch.nn.functional.pad(rir, (0, max_len - rir.shape[-1])) for rir in rirs]
    coordinates = [torch.tensor(item[1]) for item in batch]
    deformations = [torch.as_tensor(item[2]) for item in batch]
    orientations = [torch.as_tensor(item[3], dtype=torch.float64) for item in batch]
    return torch.stack(rirs), torch.stack(coordinates), torch.stack(deformations), torch.stack(orientations)