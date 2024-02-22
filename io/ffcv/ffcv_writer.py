import torch
import getpass
import os
from torch.utils.data import Dataset
import EARS.io.fast_io as fast_io
import numpy as np
import EARS.io.hdf5 as hdf5

# ffcv
from ffcv.writer import DatasetWriter
from ffcv.fields import TorchTensorField, FloatField


def shear(x, angle):
    # print(f"Shearing {x} by {angle} deg")
    angle = torch.deg2rad(angle)
    c = 1/torch.tan(angle)
    R = torch.tensor([
        [1, c],
        [0, 1],
        ])
    return R@(x.to(torch.float32))

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
                 deformation = None, shear_angle = None, filter_angles:bool=False,
                 use_newton_cluster:bool = False) -> None:
        assert deformation is not None or not return_deformation, "You need to specify the deformation"
        super().__init__()
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            raise ValueError(f"The path to the data {self.data_path} does not exist.")
        
        self.use_newton_cluster = use_newton_cluster
        if self.use_newton_cluster:
            self.extension = ".hdf5"
            user_name = getpass.getuser()
            summary_file_path_root = data_path.replace("gabrieles", user_name)
            summary_file_path_first_partition = os.path.join(summary_file_path_root, "first_partition")
            # self.first_partition = set(filter(lambda x: x.endswith(".hdf5"), os.listdir(data_path)))
            first_partition = list(filter(lambda x: x.endswith(".hdf5"), fast_io.get_listed_files(data_path, summary_file_path=summary_file_path_first_partition)))
            summary_file_path_second_partition = os.path.join(summary_file_path_root, "second_partition")
            # second_partition = set(filter(lambda x: x.endswith(".hdf5"), os.listdir(data_path.replace("gabrieles", "tamir.shor",1))))
            second_partition = list(filter(lambda x: x.endswith(".hdf5"), fast_io.get_listed_files(data_path.replace("gabrieles", "tamir.shor",1), summary_file_path=summary_file_path_second_partition)))
            self.separator = len(first_partition)
            first_partition.extend(second_partition)
            self.single_data_path = first_partition


            # #print(f"[debug] data path {data_path} second data path {data_path.replace('gabrieles', 'tamir.shor',1)}")
            # # first_partition = list(filter(lambda x: x.endswith(".hdf5"), os.listdir(data_path)))
            # first_partition = list(filter(lambda x: x.endswith(".hdf5"), fast_io.get_listed_files(data_path)))
            # # second_partition = list(filter(lambda x: x.endswith(".hdf5"), os.listdir(data_path.replace("gabrieles", "tamir.shor",1))))
            # second_partition = list(filter(lambda x: x.endswith(".hdf5"), fast_io.get_listed_files(data_path.replace("gabrieles", "tamir.shor",1))))
            # self.separator = len(first_partition)
            # #print(f"[debug] first_partition {first_partition} second partition {second_partition}")
            # first_partition.extend(second_partition)
            # self.single_data_path = first_partition
            # #print(f"[debug] single data path contains {self.single_data_path}")
        elif not filter_angles: 
        # create a list of all the files in the folder
            hdf5_datasets = set(["/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/orig_64_angles/default_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/non_convex_room/non_convex_5.0_5.0_order_1_0.5_d_0.05/"])
            self.extension = ".hdf5" if self.data_path in hdf5_datasets else ".npy"
            # self.single_data_path = list(filter(lambda x: x.endswith(self.extension), os.listdir(self.data_path)))
            self.single_data_path = list(filter(lambda x: x.endswith(self.extension), fast_io.get_listed_files(self.data_path)))
            #print(f"Working on {len(self.single_data_path)} files")
        
        #date_filter = get_filter(self.data_path)
        #self.single_data_path = list(filter(date_filter, os.listdir(self.data_path)))

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

    def __len__(self) -> int:
        return len(self.single_data_path)
    
    def get_coordinates(self, data_file:str) -> tuple:
        # print(f"processing {data_file}")
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
        # assert (all(x>=0 for x in coordinates) or all(abs(x)<1e-4 for x in coordinates)) \
        #         and (all(x<=1 for x in coordinates) or all((x-1)<=1e-4 for x in coordinates)),\
        #     f"in file {data_file} got coordinates {coordinates} in data_path {self.data_path} min {self.min_coord} max {self.max_coord}"
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
        except:
            print(f"Cannot read {os.path.join(self.data_path, data_file)}")
            raise ValueError()
        if torch.any(torch.isnan(data)):
            print(f"{os.path.join(self.data_path, data_file)} contains nan values!")
            breakpoint()
        
        # pad data to fit sound_shape
        MAX_TIME_LEN: int = 1357
        data = torch.nn.functional.pad(data, (0, MAX_TIME_LEN-data.shape[-1]))
        return data, coordinates[0], coordinates[1], orientation
        
        # old code
        if not self.return_absorption_coefficient and not self.return_deformation:
            return data, coordinates, orientation
        elif self.return_absorption_coefficient and not self.return_deformation:
            return data, coordinates, self.absorption_coefficient, orientation
        elif not self.return_absorption_coefficient and self.return_deformation:
            return data, coordinates, self.deformation, orientation
        else:
            return data, coordinates, self.absorption_coefficient, self.deformation, orientation
    
    def get_rec_len(self):
        return self.rec_len
    
data_path = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/orig_64_angles/default_5.0_5.0_order_1_0.5_d_0.05/"
min_coord: torch.Tensor = torch.tensor([0.93, 0.93])
max_coord: torch.Tensor = torch.tensor([4.07, 4.07])
dataset = Localization2dGivenOrientationDataset(data_path, duration=None, filter_angles=False, use_newton_cluster=False,
                                                            min_coord=min_coord, max_coord=max_coord)
print("dataset:")
print(dataset)
dataset = None

writer_path: str = os.path.join(data_path, "dataset.beton")
print(f"writer path: {writer_path}")
write_path: str = None

sound_shape: tuple = (4,8,1357)
sound_shape = None

writer = DatasetWriter(write_path, {
    'sound': TorchTensorField(dtype=torch.float64, shape=sound_shape),
    'x': FloatField(),
    'y': FloatField(),
    't': FloatField()
}, num_workers=40)

writer.from_indexed_dataset(dataset)