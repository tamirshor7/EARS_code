#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from EARS.localization.phase_modulation.modulation_dataset import ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient, \
    ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2d, collate_fn_absorption_coefficient, collate_fn, \
    ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2dOrientation, Localization2dGivenOrientationDataset, \
    collate_fn_absorption_coefficient_orientation, collate_fn_orientation, collate_fn_deformation_orientation
    #ModulationDatasetFixedInputSoundFixedGeometryDeformation2d, ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2dOrientation
from EARS.localization.phase_modulation.phase_modulation_pipeline import Localization_Model
import os
import sys
import argparse

# convert python dictionary to Object whose values can be accessed as attributes
from munch import Munch

from EARS.localization.multi_position import dataset as multi_position_dataset
from EARS.localization.multi_position import master, aggregator, trajectory_factory

import concurrent.futures
#signals = torch.from_numpy(np.load('signals.npy')).unsqueeze(0)
#signals = torch.from_numpy(np.load('/mnt/walkure_public/tamirs/signals.npy')).unsqueeze(0)
signals = None


losses = {
        'rms': lambda x,y: torch.sqrt(nn.MSELoss(reduction='none')(x,y)),
        'l1': nn.L1Loss(reduction='none')
    }

# norm of the direct component 
# (used as a reference to determine the magnitude of the noise in the robustness tests)
REF_NORM: float = 12.004616371442648
def linear_to_db(x, ref=REF_NORM):
    return 10*np.log10(x/ref)
def db_to_linear(x, ref=REF_NORM):
    return ref*(10**(x/10))

'''
def shear(x, angle):
    angle = torch.deg2rad(angle)
    c = 1/torch.tan(angle)
    R = torch.tensor([
        [1, c],
        [0, 1],
        ])
    return R@x
'''
def shear(x_batch, angles_batch):
    device = x_batch.device
    # Convert angles to radians
    angles_batch = torch.deg2rad(angles_batch)
    
    # Compute shear coefficients for each angle
    c_batch = 1 / torch.tan(angles_batch)
    
    # Create a batch of shear matrices
    batch_size = x_batch.shape[0]
    R_batch = torch.zeros(batch_size, 2, 2, device=device)
    R_batch[:, 0, 0] = 1
    R_batch[:, 0, 1] = c_batch
    R_batch[:, 1, 1] = 1
    
    # Apply shear transformation to each batch element
    result_batch = torch.matmul(R_batch, x_batch.unsqueeze(-1)).squeeze(-1)
    
    return result_batch

def get_min_max(room_x, room_y, margins, angle=None):
    '''
    Given room_x, room_y and the margins (in case also the angle) it returns the minimum and the maximum coordinate
    '''
    MICS_R: float = 0.9144
    if angle is not None and angle != 90:

        
        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(angle)
        
        angle = torch.deg2rad(angle)
        cotan_theta = 1/torch.tan(angle)

        # y_min = round((MICS_R+margins), 2)
        # x_min = torch.round(cotan_theta*(y_min + torch.sqrt(1+1/cotan_theta**2)*(MICS_R+margins)) - cotan_theta*y_min, decimals=2)

        # y_max = round((room_y-MICS_R-margins), 2)
        # x_max = torch.round(cotan_theta*(y_min + 1/cotan_theta*room_x-torch.sqrt(1+1/cotan_theta**2)*(MICS_R+margins)) - cotan_theta*y_min, decimals=2)
        
        x_min = torch.round(cotan_theta*torch.sqrt(1+1/cotan_theta**2)*(MICS_R+margins), decimals=2)
        y_min = round((MICS_R+margins), 2)

        x_max = torch.round(room_x - cotan_theta*torch.sqrt(1+1/cotan_theta**2)*(MICS_R+margins), decimals=2)
        y_max = round((room_y-MICS_R-margins),2)

        minimum_coordinate = torch.tensor([x_min,y_min], dtype=torch.float32)
        maximum_coordinate = torch.tensor([x_max,y_max], dtype=torch.float32)

        # R = torch.tensor([
        #     [1, -cotan_theta],
        #     [0, 1],
        #     ])
        # minimum_coordinate = (R@(minimum_coordinate.to(torch.float32)))
        # maximum_coordinate = (R@(maximum_coordinate.to(torch.float32)))
    else:
        minimum = round(MICS_R+margins, 2)
        minimum_coordinate = torch.tensor([minimum, minimum])

        maximum_x = round(room_x-MICS_R-margins, 2)
        maximum_y = round(room_y-MICS_R-margins, 2)
        maximum_coordinate = torch.tensor([maximum_x, maximum_y])


    assert all(maximum_coordinate >= minimum_coordinate), f"Error: minimum coordinate is bigger than maximum coordinate! Min {minimum_coordinate} Max {maximum_coordinate}"
    assert all(minimum_coordinate >= 0), f"Minimum coordinate cannot be negative! Got min: {minimum_coordinate} max: {maximum_coordinate} angle {angle}"
    return minimum_coordinate, maximum_coordinate

@torch.jit.script
def convolve(rir:torch.Tensor, signals:torch.Tensor):
    # rir has shape (batch_size, 8, 1024, t)
    # signals has shape         (1, 1024, t)
    # We need to break the 1024 so that we have a recording per rotor!
    reshaped_rir = rir.reshape(rir.shape[0]*rir.shape[1],rir.shape[2], rir.shape[3])
    #print(f"signals {signals.shape} reshaped_rir {reshaped_rir.shape}")
    #output_sound_sources = torch.nn.functional.conv1d(signals, reshaped_rir)
    #print(f"output_sound_sources {output_sound_sources.shape}")
    output = torch.zeros(rir.shape[0]*rir.shape[1],4, signals.shape[-1]-reshaped_rir.shape[-1]+1, dtype=torch.float64, device=rir.device)
    for i in range(4):
        #print(f"selecting {signals[:,256*i:256*(i+1)].shape} {reshaped_rir[:,256*i:256*(i+1)].shape}")
        output[:,i] = torch.nn.functional.conv1d(signals[:,256*i:256*(i+1)], torch.flip(reshaped_rir[:,256*i:256*(i+1)], [-1])).squeeze()
        #torch.sum(output_sound_sources[rir.shape[0]*rir.shape[1]*i*256:rir.shape[0]*rir.shape[1]*(i+1)*256,0,:])
    output = output.reshape(rir.shape[0], rir.shape[1], 4, output.shape[-1])
    output = torch.permute(output, (0,2,1,3))
    #breakpoint()
    #print(f"output {output.shape}")
    return output

# temporary: to remove!
class ModulationDatasetFixedInputSoundFixedGeometryDeformation2d(torch.utils.data.Dataset):
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
                                                        shear_angle:float = None) -> None:
        # shear_angle is the angle used to shear the room
        super().__init__()
        self.rir_data_path = rir_data_path
        assert rir_data_path is not None, "You need to specify the path, please set rir_data_path"
        if not os.path.exists(self.rir_data_path):
            raise ValueError(f"The path to the rir data {self.rir_data_path} does not exist.")
        # create a list of all the files in the folder
        self.single_rir_path = os.listdir(self.rir_data_path)
        input_sound_parameters = np.load(os.path.join(os.path.dirname(__file__),  "..", "dataset", "input_sound_bank", "input_sound_parameters.npz"))
        self.rec_len = int(input_sound_parameters["duration"]*input_sound_parameters["fs"])
        self.deformation = deformation
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.min_max_scaling = min_max_scaling
        self.return_deformation = return_deformation
        # if shear_angle is not None:
        #     self.shear_angle = torch.deg2rad(torch.tensor(-shear_angle))
        #     c = 1/torch.tan(self.shear_angle)
        #     self.R = torch.tensor([
        #         [1, c],
        #         [0, 1],
        #         ])
        # else:
        #     self.shear_angle = None

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

        # if self.shear_angle is not None:
        #     coordinates = torch.tensor(coordinates)
        #     coordinates = (self.R@(coordinates.to(torch.float32)))
        #     coordinates = tuple(coordinates)

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


def geometric_median(points, iterations=20, epsilon=1e-10):
    y = np.mean(points, axis=0, keepdims=True)
    for _ in range(iterations):
        distances = np.linalg.norm(points-y, axis=-1, keepdims=True)
        weights = 1.0 / (distances+epsilon)
        y = np.sum(weights*points, axis=0, keepdims=True)/np.sum(weights, axis=0, keepdims=True)
    y = np.squeeze(y)
    return y

def get_geometric_median_loss(df: pd.DataFrame) -> pd.Series:
    geometric_median_coordinates: np.ndarray = np.concatenate(df['geometric_median'].values).reshape(-1,2)
    ground_truth_coordinate: np.ndarray = np.concatenate((
                np.expand_dims(df['ground_truth_coordinate_0'].values, -1), 
                np.expand_dims(df['ground_truth_coordinate_1'].values, -1)),
                axis=-1)
    loss: np.ndarray = np.sqrt(np.mean((geometric_median_coordinates-ground_truth_coordinate)**2, axis=-1))
    loss: pd.Series = pd.Series(loss, name="geometric_median_loss")
    return loss



def evaluate(model: nn.Module, dataset: torch.utils.data.Dataset, perturbation: np.ndarray, 
             batch_size: int = 32, loss_function: str = 'rms', 
             dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> pd.DataFrame:
    '''Evaluate the model on the dataset.

    :arg model: the model to evaluate
    :arg dataset: the dataset to evaluate on (torch.utils.data.Dataset)
    :arg batch_size: the batch size to use when evaluating
    :arg loss_fn: the loss function to use among 'mse' and 'l1'
    :arg dev: the device to use for evaluation
    :arg perturbation: the perturbation that generated the datapoints in the dataset
    :returns: a pandas dataframe containing the evaluation metrics
    '''
    # type checking
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert isinstance(batch_size, int)
    assert loss_function in ['rms', 'l1']
    assert isinstance(dev, torch.device)
    assert isinstance(perturbation, np.ndarray)

    assert len(dataset) == len(perturbation), 'dataset and perturbation must have the same length'

    loss_fn = losses[loss_function]

    # instantiate dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # evaluate
    model.eval()
    model = model.to(dev)
    loss = []
    predicted = []
    with tqdm(total=len(dataloader)) as pbar:
        pbar.set_description('Evaluating')
        with torch.no_grad():
            for batch in dataloader:
                if not dataset.fixed_phase:
                    x,y = batch
                    x = x.to(dev)
                    y = y.to(dev)
                    y_hat = model(x)
                else:
                    x, phi, y = batch
                    x = x.to(dev)
                    phi = phi.to(dev)
                    y = y.to(dev)
                    y_hat = model(x, phi)
                loss.extend(loss_fn(y_hat, y).cpu().numpy().tolist())
                predicted.extend(y_hat.cpu().numpy().tolist())
                pbar.update(1)

    # compute metrics
    loss = np.array(loss)
    #if loss_function == 'rms':
        #loss = np.sqrt(loss)
    
    df = pd.DataFrame({'perturbation': perturbation, 'loss': loss, 'predicted': predicted})
    return df

def evaluate_absorption_coefficient_robustness(model: nn.Module, batch_size:int = 10,
                                               loss_function: str = 'rms',
                                               dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> pd.DataFrame:
    '''Evaluate the model on the dataset varying the absorption coefficient.

    :arg model: the model to evaluate
    :arg batch_size: the batch size to use when evaluating
    :arg loss_fn: the loss function to use among 'mse' and 'l1'
    :arg dev: the device to use for evaluation
    :returns: a pandas dataframe containing the evaluation metrics
    '''
    # type checking
    assert isinstance(batch_size, int)
    assert loss_function in ['rms', 'l1']
    assert isinstance(dev, torch.device)

    dataset = torch.utils.data.ConcatDataset([
        ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient(absorption_coefficient=absorption_coefficient, step_distance=0.1)
        for absorption_coefficient in np.arange(0.05, 0.8, 0.05)
    ])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    loss_fn = losses[loss_function]

    # evaluate
    model.eval()
    model = model.to(dev)
    # df will contain a list of tuples (absorption_coefficient, ground_truth_distance, predicted_distance, loss)
    df = []
    with tqdm(total=len(dataloader)) as pbar:
        pbar.set_description('Evaluating')
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(dev)
                y = y.to(dev)
                y_hat = model(x)
                df.extend(
                    list(zip(
                        x[:, 0].cpu().numpy().tolist(), # absorption coefficient
                        y.cpu().numpy().tolist(), # ground truth distance
                        y_hat.cpu().numpy().tolist(), # predicted distance
                        loss_fn(y_hat, y).cpu().numpy().tolist() # loss
                    ))
                )
                pbar.update(1)
    
    df = pd.DataFrame(df, columns=['absorption_coefficient', 'ground_truth_distance', 'predicted_distance', 'loss'])
    # save dataframe
    # check if file with this name exists
    counter = 0
    name = f'absorption_coefficient_robustness_{counter}.csv'
    while os.path.exists(name):
        counter += 1
        name = f'absorption_coefficient_robustness_{counter}.csv'
    df.to_csv(name, index=False)
    print(f"Dataframe saved to {name}")
    return df

def evaluate_absorption_coefficient_robustness_2d(model: nn.Module, batch_size:int = 10,
                                               loss_function: str = 'rms',
                                               dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                               experiment_name: str = None,
                                               testing_absorption_coefficients:list = None) -> pd.DataFrame:
    '''Evaluate the model on the dataset varying the absorption coefficient.

    :arg model: the model to evaluate
    :arg batch_size: the batch size to use when evaluating
    :arg loss_fn: the loss function to use among 'mse' and 'l1'
    :arg dev: the device to use for evaluation
    :returns: a pandas dataframe containing the evaluation metrics
    '''
    # type checking
    assert isinstance(batch_size, int)
    assert loss_function in ['rms', 'l1']
    assert isinstance(dev, torch.device)

    print('Running absorption coefficient test...')

    # create_name = lambda absorption_coefficient: f"rel_units_rir_indoor_absorption_coefficient_5.0_5.0_order_1_{round(absorption_coefficient,2)}_d_0.05"
    create_name = lambda absorption_coefficient: f"e_{round(absorption_coefficient,2):.2f}_5.0_5.0_order_1_{round(absorption_coefficient,2)}_d_0.05"

    if testing_absorption_coefficients is not None and len(testing_absorption_coefficients) > 0:
        coefficients = testing_absorption_coefficients
    else:
        coefficients = np.arange(0.05, 1.0, 0.05)

    dataset = torch.utils.data.ConcatDataset([
        # rir_indoor_0dot75_5.0_5.0_order_1_0.75_d_0.44
        # rir_absorption_coefficient_5.0_5.0_order_1_0.05_d_0.05
        #ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2d(absorption_coefficient=absorption_coefficient,
        Localization2dGivenOrientationDataset(absorption_coefficient=absorption_coefficient, 
                                                                    #  rir_data_path=os.path.join('/mnt', 'walkure_public', 'tamirs','rir_robustness_test', 'absorption_coefficient',
                                                                    # data_path=os.path.join('/mnt', 'walkure_public', 'tamirs','pressure_field_orientation_dataset', 'robustness_test',
                                                                    data_path=os.path.join('/mnt', 'walkure_public', 'tamirs','pressure_field_orientation_dataset', 'absorption_coefficient',
        # create_name(absorption_coefficient=absorption_coefficient),"rir", "rir_indoor"),
        create_name(absorption_coefficient=absorption_coefficient)),
                                                                     return_absorption_coefficient=True)
        for absorption_coefficient in coefficients
    ])
    # print("ATTENTION: We're only testing the absorption coefficient 0.5 now!")
    # dataset = Localization2dGivenOrientationDataset(absorption_coefficient=0.5,
    #     data_path=os.path.join('/mnt', 'walkure_public', 'tamirs','pressure_field_orientation_dataset', '16_dataset',
    #                            'rel_units_rir_indoor_absorption_coefficient_5.0_5.0_order_1_0.5_d_0.05',
    #                            ),
    #                                                                  return_absorption_coefficient=True)

    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn_absorption_coefficient)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, collate_fn=collate_fn_absorption_coefficient_orientation,
                                             num_workers=10)

    loss_fn = losses[loss_function]

    # evaluate
    model.eval()
    model = model.to(dev)
    # df will contain a list of tuples (absorption_coefficient, ground_truth_distance, predicted_distance, loss)
    df = []
    with tqdm(total=len(dataloader)) as pbar:
        pbar.set_description('Evaluating')
        with torch.no_grad():
            for batch in dataloader:
                x, y, absorption_coefficients, orientation = batch
                #x = convolve(x.to(dev), signals=signals)
                x = x.to(dev)
                y = y.to(dev)
                orientation = orientation.to(dev)
                y_hat = model(x, orientation=orientation)
                df.extend(
                    list(zip(
                        absorption_coefficients.cpu().numpy().tolist(), # absorption coefficient
                        y.cpu().numpy().tolist(), # ground truth coordinate
                        y_hat.cpu().numpy().tolist(), # predicted coordinate
                        loss_fn(y_hat, y).cpu().numpy().tolist(), # loss
                        orientation.cpu().numpy().tolist()
                    ))
                )
                pbar.update(1)
    
    df = pd.DataFrame(df, columns=['absorption_coefficient', 'ground_truth_coordinate', 'predicted_coordinate', 'loss', 'angle'])
    # save dataframe
    # check if file with this name exists
    counter = 0
    name = f'2d_absorption_coefficient_robustness_{counter}_{experiment_name}_{coefficients}.csv'
    while os.path.exists(name):
        counter += 1
        name = f'2d_absorption_coefficient_robustness_{counter}_{experiment_name}_{coefficients}.csv'
    df.to_csv(name, index=False)
    print(f"Dataframe saved to {name}")
    return df

def evaluate_phase_modulation_robustness(model: nn.Module, batch_size:int = 10,
                                               loss_function: str = 'rms',
                                               dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> pd.DataFrame:
    '''Evaluate the model on the dataset varying the absorption coefficient.

    :arg model: the model to evaluate
    :arg batch_size: the batch size to use when evaluating
    :arg loss_fn: the loss function to use among 'mse' and 'l1'
    :arg dev: the device to use for evaluation
    :returns: a pandas dataframe containing the evaluation metrics
    '''
    # type checking
    assert isinstance(batch_size, int)
    assert loss_function in ['rms', 'l1']
    assert isinstance(dev, torch.device)

    dataset = ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient(absorption_coefficient=0.2, step_distance=0.1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    loss_fn = losses[loss_function]

    # evaluate
    model.eval()
    model = model.to(dev)
    # df will contain a list of tuples (standard_deviation, ground_truth_distance, predicted_distance, loss)
    df = []
    n_experiments = 3
    standard_deviations = [1e-4, 1e-3, 1e-2, 1e-1]
    rng = np.random.default_rng()
    for standard_deviation in standard_deviations:
        print(f"Processing standard deviation: {standard_deviation}")
        with tqdm(total=len(dataloader)) as pbar:
            pbar.set_description('Evaluating')
            with torch.no_grad():
                for batch in dataloader:
                    x, y = batch
                    x = x.to(dev)
                    y = y.to(dev)
                    avg_loss = 0
                    avg_predicted_distance = 0
                    for i in range(n_experiments):
                        injected_phase = model.phase_model()
                        injected_phase += torch.as_tensor(rng.normal(scale=standard_deviation, size=injected_phase.shape)).to(injected_phase.device, injected_phase.dtype)
                        y_hat = model(x, injected_phases=injected_phase)
                        loss = loss_fn(y_hat, y).cpu().numpy()
                        avg_loss = (i*avg_loss+loss)/(i+1)
                        avg_predicted_distance = (i*y_hat+y_hat)/(i+1)

                    df.extend(
                        list(zip(
                            [standard_deviation]*avg_loss.shape[0], # standard deviation
                            y.cpu().numpy().tolist(), # ground truth distance
                            avg_predicted_distance.cpu().numpy().tolist(), # predicted distance
                            avg_loss.tolist() # loss
                        ))
                    )
                    pbar.update(1)
    
    df = pd.DataFrame(df, columns=['standard_deviation','ground_truth_distance', 'predicted_distance', 'loss'])
    # save dataframe
    # check if file with this name exists
    counter = 0
    name = f'phase_modulation_robustness_{counter}.csv'
    while os.path.exists(name):
        counter += 1
        name = f'phase_modulation_robustness_{counter}.csv'
    df.to_csv(name, index=False)
    print(f"Dataframe saved to {name}")
    return df

def evaluate_input_noise_robustness(model: nn.Module, batch_size:int = 10, 
                              loss_function: str = 'rms',
                            dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                            experiment_name:str = None,
                            use_newton_cluster:bool = False, check_training_set:bool = False,
                            system_noise_gain:float=1e-1,
                            picked_noise_intensities:list = None) -> pd.DataFrame:
    '''Evaluate the model on the dataset varying the absorption coefficient.

    :arg model: the model to evaluate
    :arg batch_size: the batch size to use when evaluating
    :arg loss_fn: the loss function to use among 'mse' and 'l1'
    :arg dev: the device to use for evaluation
    :returns: a pandas dataframe containing the evaluation metrics
    '''
    # type checking
    assert isinstance(batch_size, int)
    assert loss_function in ['rms', 'l1']
    assert isinstance(dev, torch.device)

    #DATA_PATH:str = '/home/gabriele/EARS_project/data/rir/rir_indoor_4_channels_5.0_5.0_order_1_0.5_d_0.05/rir_indoor/'
    #DATA_PATH:str = '/mnt/walkure_public/tamirs/rir2d/rir_indoor/'
    #DATA_PATH:str = '/mnt/walkure_public/tamirs/pressure_field_2d_no_padding/indoor_recordings_4_rotors_8_mics_d_0.05_mode_indoor_None/'
    #DATA_PATH:str = '/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/default_5.0_5.0_order_1_0.5_d_0.05'
    if use_newton_cluster:
        DATA_PATH: str = "/home/gabrieles/EARS/data/pressure_field_orientation_dataset/32_angles/default_5.0_5.0_order_1_0.5_d_0.05/"
    else:
        DATA_PATH:str = '/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/orig_64_angles/default_5.0_5.0_order_1_0.5_d_0.05/'
    #dataset = ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2d(DATA_PATH)
    dataset = Localization2dGivenOrientationDataset(DATA_PATH, use_newton_cluster=use_newton_cluster)
    # use 80 train, 10% val, 10% test
    train_split_ratio = 0.8
    val_split_ratio = 0.10
    if check_training_set:
        train_dataset, _, test_dataset = torch.utils.data.random_split(dataset,
                                                                                [int(train_split_ratio * len(dataset)),
                                                                                int(val_split_ratio * len(dataset)),
                                                                                len(dataset) - int(
                                                                                    train_split_ratio * len(
                                                                                        dataset)) - int(
                                                                                    val_split_ratio * len(dataset))])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_orientation, pin_memory=True)

    else:
        _, _, test_dataset = torch.utils.data.random_split(dataset,
                                                                                [int(train_split_ratio * len(dataset)),
                                                                                int(val_split_ratio * len(dataset)),
                                                                                len(dataset) - int(
                                                                                    train_split_ratio * len(
                                                                                        dataset)) - int(
                                                                                    val_split_ratio * len(dataset))])
    #train_subset = torch.utils.data.Subset(train_dataset, range(0, len(train_dataset), 10)) 

    #val_subset = torch.utils.data.Subset(val_dataset, range(0, len(val_dataset), 10))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_orientation, pin_memory=True,
                                                  num_workers=5)

    loss_fn = losses[loss_function]

    # evaluate
    model.eval()
    model = model.to(dev)
    # df will contain a list of tuples (standard_deviation, ground_truth_distance, predicted_distance, loss, angle)
    df = []
    n_experiments = 3
    if picked_noise_intensities is not None and len(picked_noise_intensities)> 0:
        noise_intensities = picked_noise_intensities
    else:
        noise_intensities = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75, 'inf']
    print(f"checking {noise_intensities} noise levels...")
    #noise_intensities = ['inf']
    # Test robustness in training dataset
    if check_training_set:
        for noise_intensity in noise_intensities:
            print(f"Processing noise intensity: {noise_intensity}")
            with tqdm(total=len(train_dataloader)) as pbar:
                pbar.set_description('Evaluating')
                with torch.no_grad():
                    for batch in train_dataloader:
                        x, y, orientation = batch
                        x = x.to(dev)
                        y = y.to(dev)
                        orientation = orientation.to(dev)
                        avg_loss = 0
                        avg_predicted_distance = 0
                        for i in range(n_experiments):
                            # y_hat = model(x, norm_noise_injected_in_sound=db_to_linear(noise_intensity))
                            if noise_intensity == 'inf':
                                y_hat = model(x, orientation=orientation)
                            else:
                                y_hat = model(x, orientation=orientation, desired_snr_in_db=noise_intensity)
                            loss = loss_fn(y_hat, y).cpu().numpy()
                            avg_loss = (i*avg_loss+loss)/(i+1)
                            avg_predicted_distance = (i*avg_predicted_distance+y_hat)/(i+1)

                        df.extend(
                            list(zip(
                                [noise_intensity]*avg_loss.shape[0], # intensity of the noise
                                y.cpu().numpy().tolist(), # ground truth coordinate
                                avg_predicted_distance.cpu().numpy().tolist(), # predicted coordinate
                                avg_loss.tolist(), # loss
                                [True]*avg_loss.shape[0], # True signals that we are working with the training dataset
                                [orientation]*avg_loss.shape[0],
                            ))
                        )
                        pbar.update(1)
    # Test robustness in validation dataset
    for noise_intensity in noise_intensities:
        print(f"Processing noise intensity: {noise_intensity}")
        with tqdm(total=len(test_dataloader)) as pbar:
            pbar.set_description('Evaluating')
            with torch.no_grad():
                for batch in test_dataloader:
                    x, y_cpu, orientation_cpu = batch
                    x = x.to(dev)
                    y = y_cpu.to(dev)
                    orientation = orientation_cpu.to(dev)
                    avg_loss = 0
                    avg_predicted_distance = 0
                    for i in range(n_experiments):
                        if noise_intensity == 'inf':
                            y_hat = model(x, orientation=orientation)
                        else:
                            y_hat = model(x, orientation=orientation, desired_snr_in_db=noise_intensity)
                        loss = loss_fn(y_hat, y).cpu().numpy()
                        avg_loss = (i*avg_loss+loss)/(i+1)
                        avg_predicted_distance = (i*avg_predicted_distance+y_hat)/(i+1)

                    df.extend(
                        list(zip(
                            [noise_intensity]*avg_loss.shape[0], # intensity of the noise
                            y_cpu.numpy().tolist(), # ground truth coordinate
                            avg_predicted_distance.cpu().numpy().tolist(), # predicted coordinate
                            avg_loss.tolist(), # loss
                            [False]*avg_loss.shape[0], # False signals that we are working with the validation dataset
                            orientation_cpu.numpy().tolist(),
                        ))
                    )
                    pbar.update(1)
    
    df = pd.DataFrame(df, columns=['snr','ground_truth_coordinate', 'predicted_coordinate', 'loss', 'is_training_dataset', 'orientation'])
    counter = 0
    if experiment_name is not None:
        name = f'noise_in_input_robustness_{counter}_{experiment_name}_noise_intensities_{noise_intensities}.csv'
    else:
        name = f'noise_in_input_robustness_{counter}_noise_intensities_{noise_intensities}.csv'
    while os.path.exists(name):
        counter += 1
        if experiment_name is not None:
            name = f'noise_in_input_robustness_{counter}_{experiment_name}_noise_intensities_{noise_intensities}.csv'
        else:
            name = f'noise_in_input_robustness_{counter}_noise_intensities_{noise_intensities}.csv'
    raw_name = f"{name.removesuffix('.csv')}_raw.csv"
    df.to_csv(raw_name, index=False)
    print(f"Raw dataframe saved to {raw_name}")
    
    df['ground_truth_coordinate_0'] = df['ground_truth_coordinate'].apply(lambda x: x[0]).astype(float)
    df['ground_truth_coordinate_1'] = df['ground_truth_coordinate'].apply(lambda x: x[1]).astype(float)
    df['predicted_coordinate_0'] = df['predicted_coordinate'].apply(lambda x: x[0]).astype(float)
    df['predicted_coordinate_1'] = df['predicted_coordinate'].apply(lambda x: x[1]).astype(float)
    df['loss_0'] = df['loss'].apply(lambda x: x[0]).astype(float)
    df['loss_1'] = df['loss'].apply(lambda x: x[1]).astype(float)
    df['loss'] = (df['loss_0'] + df['loss_1']) / 2

    df = df.drop(columns=['ground_truth_coordinate', 'predicted_coordinate', 'loss_0', 'loss_1'])
    if not check_training_set:
        df = df.drop(columns='is_training_dataset')
    # save dataframe
    # check if file with this name exists
    df.to_csv(name, index=False)
    print(f"Dataframe saved to {name}")

    if check_training_set:
        df_summary_mean = df[df.is_training_dataset == False].groupby('snr')['loss'].mean()
        df_summary_std = df[df.is_training_dataset == False].groupby('snr')['loss'].std()
    else:
        df_summary_mean = df.groupby('snr')['loss'].mean()
        df_summary_std = df.groupby('snr')['loss'].std()
    df_summary = pd.merge(df_summary_mean, df_summary_std, on='snr')
    #df_summary.to_csv(f'noise_in_input_robustness_0_summary.csv')
    df_summary.rename(columns={'loss_x':'mean', 'loss_y':'std'}, inplace=True)

    if experiment_name is not None:
        summary_name = f'noise_in_input_robustness_{counter}_{experiment_name}_summary_noise_intensities_{noise_intensities}.csv'
    else:
        summary_name = f'noise_in_input_robustness_{counter}_summary_noise_intensities_{noise_intensities}.csv'
    df_summary.to_csv(summary_name)
    print(f"Summary dataframe saved to {summary_name}")
    return df


def evaluate_non_uniform_deformation_robustness(model: nn.Module, batch_size:int = 10, 
                              loss_function: str = 'rms',
                            dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                            experiment_name:str = None) -> pd.DataFrame:
    '''Evaluate the model on the dataset varying the side of the room.

    :arg model: the model to evaluate
    :arg batch_size: the batch size to use when evaluating
    :arg loss_fn: the loss function to use among 'mse' and 'l1'
    :arg dev: the device to use for evaluation
    :returns: a pandas dataframe containing the evaluation metrics
    '''
    # type checking
    assert isinstance(batch_size, int)
    assert loss_function in ['rms', 'l1']
    assert isinstance(dev, torch.device)         

    create_name = lambda deformation: f"rir_indoor_0dot05_non_uniform_{str(round(deformation*5.0,1))}_5.0_order_1_0.5_d_0.44"

    dataset = torch.utils.data.ConcatDataset([
        # rir_indoor_0dot05_non_uniform_10.0_5.0_order_1_0.5_d_0.44
        ModulationDatasetFixedInputSoundFixedGeometryDeformation2d(deformation=deformation, 
                                                                     rir_data_path=os.path.join(os.path.dirname(__file__), 
        "..", "..", "data", "rir", create_name(deformation=deformation),"rir_indoor"),
                                                                     return_deformation=True)
        for deformation in np.arange(0.5, 2.0, 0.1)
    ])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn_absorption_coefficient)

    loss_fn = losses[loss_function]

    # evaluate
    model.eval()
    model = model.to(dev)
    # df will contain a list of tuples (deformation, ground_truth_distance, predicted_distance, loss)
    df = []
    with tqdm(total=len(dataloader)) as pbar:
        pbar.set_description('Evaluating')
        with torch.no_grad():
            for batch in dataloader:
                x, y, deformation = batch
                x = x.to(dev)
                y = y.to(dev)
                y_hat = model(x)
                df.extend(
                    list(zip(
                        deformation.cpu().numpy().tolist(), # deformation
                        y.cpu().numpy().tolist(), # ground truth coordinate
                        y_hat.cpu().numpy().tolist(), # predicted coordinate
                        loss_fn(y_hat, y).cpu().numpy().tolist() # loss
                    ))
                )
                pbar.update(1)
    
    df = pd.DataFrame(df, columns=['single_wall_deformation', 'ground_truth_coordinate', 'predicted_coordinate', 'loss'])
    # save dataframe
    # check if file with this name exists
    counter = 0
    name = f'2d_non_uniform_robustness_{counter}.csv'
    while os.path.exists(name):
        counter += 1
        name = f'2d_non_uniform_robustness_{counter}.csv'
    df.to_csv(name, index=False)
    print(f"Dataframe saved to {name}")
    return df


def evaluate_uniform_deformation_robustness(model: nn.Module, batch_size:int = 10, 
                              loss_function: str = 'rms',
                            dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                            experiment_name:str = None,
                            system_noise_gain:float = 1e-1) -> pd.DataFrame:
    '''Evaluate the model on the dataset varying the side of the room.

    :arg model: the model to evaluate
    :arg batch_size: the batch size to use when evaluating
    :arg loss_fn: the loss function to use among 'mse' and 'l1'
    :arg dev: the device to use for evaluation
    :returns: a pandas dataframe containing the evaluation metrics
    '''
    # type checking
    assert isinstance(batch_size, int)
    assert loss_function in ['rms', 'l1']
    assert isinstance(dev, torch.device)         

    '''
    dataset = torch.utils.data.ConcatDataset([
        # rir_square_2.18_2.18_order_1_0.5_d_0.05
        ModulationDatasetFixedInputSoundFixedGeometryDeformation2d(deformation=deformation, 
                                                                     rir_data_path=os.path.join(os.path.dirname(__file__), 
        "..", "..", "data", "rir", create_name(deformation=deformation),"rir_indoor"),
                                                                     return_deformation=True)
        for deformation in np.arange(0.5, 2.0, 0.05)
    ])
    '''

    #rir_constant_factor_4.705208_4.705208_order_1_0.5_d_0.05
    get_side = lambda deformation: round(np.sqrt(deformation)*5,6) if deformation>=1 else round((np.sqrt(deformation) +((2.0*(1-np.sqrt(deformation))*0.9144+2.0*0.02)/5.0))*5.0, 6)
    # create_name = lambda deformation: f"rir_constant_factor_{get_side(deformation)}_{get_side(deformation)}_order_1_0.5_d_0.05"
    # create_name = lambda deformation: f"rel_units_rir_uniform_new_deformation_{get_side(deformation)}_{get_side(deformation)}_order_1_0.5_d_0.05"
    create_name = lambda deformation: f"u_{deformation:.2f}_{get_side(deformation)}_{get_side(deformation)}_order_1_0.5_d_0.05"
    get_margin = lambda deformation: round(np.sqrt(deformation)*0.02 + (np.sqrt(deformation)-1)*0.9144, 6) if deformation>=1 else round(0.02,6)

    
    datasets = []
    for deformation in np.arange(0.5, 2.05, 0.05):
        name = create_name(deformation=deformation)
        #rir_data_path = os.path.join("/","home", "gabriele", "EARS_project", "data", "rir", create_name(deformation=deformation),"rir_indoor")
        #/mnt/walkure_public/tamirs/rir_robustness_test/uniform_deformation/
        # rir_data_path = os.path.join("/","mnt", "walkure_public", "tamirs", "rir_robustness_test", "uniform_deformation", create_name(deformation=deformation),"rir_indoor")
        # rir_data_path = os.path.join("/","mnt", "walkure_public", "tamirs", "pressure_field_orientation_dataset", "robustness_test", name) #,"rir","rir_indoor")
        rir_data_path = os.path.join("/","mnt", "walkure_public", "tamirs", "pressure_field_orientation_dataset", "uniform", name) #,"rir","rir_indoor")
        room_x = get_side(deformation)
        room_y = get_side(deformation)
        margins = get_margin(deformation)
        minimum_coordinate, maximum_coordinate = get_min_max(room_x, room_y, margins)
        #single_dataset = ModulationDatasetFixedInputSoundFixedGeometryDeformation2d(
        single_dataset = Localization2dGivenOrientationDataset(
            deformation=deformation, data_path=rir_data_path, return_deformation=True, 
            min_coord=minimum_coordinate, max_coord=maximum_coordinate,
            skip_bad_files=True
        )
        datasets.append(single_dataset)
    dataset = torch.utils.data.ConcatDataset(datasets)
    
    # deformation = 2.0
    # name = create_name(deformation=deformation)
    # rir_data_path = os.path.join("/","home", "gabriele", "EARS_project", "data", "rir", create_name(deformation=deformation),"rir_indoor")
    # room_x = get_side(deformation)
    # room_y = get_side(deformation)
    # margins = get_margin(deformation)
    # minimum_coordinate, maximum_coordinate = get_min_max(room_x, room_y, margins)
    # dataset = ModulationDatasetFixedInputSoundFixedGeometryDeformation2d(
    #     deformation=deformation, rir_data_path=rir_data_path, return_deformation=True, 
    #     min_coord=minimum_coordinate, max_coord=maximum_coordinate
    # )


    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn_absorption_coefficient)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, collate_fn=collate_fn_deformation_orientation,
                        num_workers=10)

    loss_fn = losses[loss_function]

    # evaluate
    model.eval()
    model = model.to(dev)
    # df will contain a list of tuples (deformation, ground_truth_distance, predicted_distance, loss)
    df = []
    with tqdm(total=len(dataloader)) as pbar:
        pbar.set_description('Evaluating')
        with torch.no_grad():
            for batch in dataloader:
                x, y_cpu, deformation, orientation_cpu = batch
                if x is None or y_cpu is None or deformation is None or orientation_cpu is None:
                    continue
                x = x.to(dev)
                #x = convolve(x.to(dev), signals=signals)
                y = y_cpu.to(dev)
                orientation = orientation_cpu.to(dev)
                y_hat = model(x, orientation=orientation)
                df.extend(
                    list(zip(
                        deformation.cpu().numpy().tolist(), # deformation
                        y_cpu.numpy().tolist(), # ground truth coordinate
                        y_hat.cpu().numpy().tolist(), # predicted coordinate
                        loss_fn(y_hat, y).cpu().numpy().tolist(), # loss
                        orientation_cpu.numpy().tolist() # orientation
                    ))
                )
                pbar.update(1)
    
    df = pd.DataFrame(df, columns=['uniform_deformation', 'ground_truth_coordinate', 'predicted_coordinate', 'loss', 'angle'])
    # save dataframe
    # check if file with this name exists
    counter = 0
    name = f'2d_uniform_robustness_{counter}_{experiment_name}.csv'
    while os.path.exists(name):
        counter += 1
        name = f'2d_uniform_robustness_{counter}_{experiment_name}.csv'
    df.to_csv(name, index=False)
    print(f"Dataframe saved to {name}")
    return df



def evaluate_angle_deformation_robustness(model: nn.Module, batch_size:int = 10, 
                              loss_function: str = 'rms',
                            dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                            experiment_name:str = '',
                            save_every_n_iterations:int = 10) -> pd.DataFrame:
    '''Evaluate the model on the dataset varying the side of the room.

    :arg model: the model to evaluate
    :arg batch_size: the batch size to use when evaluating
    :arg loss_fn: the loss function to use among 'mse' and 'l1'
    :arg dev: the device to use for evaluation
    :returns: a pandas dataframe containing the evaluation metrics
    '''
    # type checking
    assert isinstance(batch_size, int)
    assert loss_function in ['rms', 'l1']
    assert isinstance(dev, torch.device)         

    #path = '/datasets/rir_indoor'
    path = '/mnt/walkure_public/tamirs/rir_orientation_angle/rir_indoor_4_channels_orientation_5.0_5.0_order_1_0.5_d_0.05_d_angle_0.39269908169872414/rir_indoor/'

    dataset = ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2dOrientation(
        rir_data_path=path, 
    )
    
    print(f'Length of the dataset: {len(dataset)}')
    loss_fn = losses[loss_function]

    # evaluate
    model.eval()
    model = model.to(dev)

    # add checkpointing logic
    # check if file with this name exists
    counter = 0
    name = f'2d_angle_robustness_{experiment_name}_{counter}.csv'
    if not os.path.exists(name):
        df = pd.DataFrame(columns=['angle', 'ground_truth_coordinate', 'predicted_coordinate', 'loss'])
        current_iteration = 0
    else:
        # Take the latest version of the file
        while os.path.exists(name):
            counter += 1
            name = f'2d_angle_robustness_{experiment_name}_{counter}.csv'
        df = pd.read_csv(f'2d_angle_robustness_{experiment_name}_{counter-1}.csv')
        current_iteration = len(df)

    #resume_batch_index = current_iteration // batch_size
    dataset = torch.utils.data.Subset(dataset, range(current_iteration, len(dataset)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    # df will contain a list of tuples (deformation, ground_truth_distance, predicted_distance, loss)
    #df = []
    with tqdm(total=len(dataloader)) as pbar:
        pbar.set_description('Evaluating')
        with torch.no_grad():
            for index, batch in enumerate(dataloader):
                x, y = batch
                x = x.to(dev)
                y = y.to(dev)
                angle = y[:, -1]
                y = y[:, :-1]
                y_hat = model(x)
                '''
                df.extend(
                    list(zip(
                        angle.cpu().numpy().tolist(), # angle
                        y.cpu().numpy().tolist(), # ground truth coordinate
                        y_hat.cpu().numpy().tolist(), # predicted coordinate
                        loss_fn(y_hat, y).cpu().numpy().tolist() # loss
                    ))
                )
                '''
                data_to_insert = list(zip(
                        angle.cpu().numpy().tolist(), # angle
                        y.cpu().numpy().tolist(), # ground truth coordinate
                        y_hat.cpu().numpy().tolist(), # predicted coordinate
                        loss_fn(y_hat, y).cpu().numpy().tolist() # loss
                    ))
                df = df.append(pd.DataFrame(data_to_insert, columns=df.columns), ignore_index=True)
                if index % save_every_n_iterations == 0:
                    df.to_csv(name, index=False)

                pbar.update(1)
    '''
    df = pd.DataFrame(df, columns=['angle', 'ground_truth_coordinate', 'predicted_coordinate', 'loss'])
    # save dataframe
    # check if file with this name exists
    counter = 0
    name = f'2d_angle_robustness_{counter}.csv'
    while os.path.exists(name):
        counter += 1
        name = f'2d_angle_robustness_{counter}.csv'
    '''
    df.to_csv(name, index=False)
    print(f"Dataframe saved to {name}")
    return df



def evaluate_aspect_ratio_deformation_robustness(model: nn.Module, batch_size:int = 10, 
                              loss_function: str = 'rms',
                            dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                            experiment_name:str = None,
                            system_noise_gain:float = 1e-1) -> pd.DataFrame:
    '''Evaluate the model on the dataset varying the side of the room.

    :arg model: the model to evaluate
    :arg batch_size: the batch size to use when evaluating
    :arg loss_fn: the loss function to use among 'mse' and 'l1'
    :arg dev: the device to use for evaluation
    :returns: a pandas dataframe containing the evaluation metrics
    '''
    # type checking
    assert isinstance(batch_size, int)
    assert loss_function in ['rms', 'l1']
    assert isinstance(dev, torch.device)         

    #create_name = lambda deformation: f"rir_aspect_ratio_{deformation:.2f}_{round(np.sqrt(1+deformation)*5,2)}_{round(5/np.sqrt(1+deformation),2)}_order_1_0.5_d_0.05"
    # create_name = lambda deformation: f"rel_units_rir_aspect_ratio_{round(np.sqrt(1+deformation)*5,2)}_{round(5/np.sqrt(1+deformation),2)}_order_1_0.5_d_0.05"
    create_name = lambda deformation: f"asp_{deformation:.2f}_{round(np.sqrt(1+deformation)*5,2)}_{round(5/np.sqrt(1+deformation),2)}_order_1_0.5_d_0.05"
    
    datasets = []
    #for deformation in np.arange(-0.85, 0.95, 0.05):
    for deformation in np.arange(-0.5, 0.95, 0.05):
        if np.abs(deformation)<1e-4:
            deformation = 0.0
        name = create_name(deformation=deformation)
        #rir_data_path = os.path.join("/","home", "gabriele", "EARS_project", "data", "rir", create_name(deformation=deformation),"rir_indoor")
        # rir_data_path = os.path.join("/mnt","walkure_public", "tamirs", "rir2d_aspect_ratio_again", create_name(deformation=deformation),"rir_indoor")
        #rir_data_path = os.path.join("/mnt","walkure_public", "tamirs", "rir_robustness_test","aspect_ratio", create_name(deformation=deformation),"rir_indoor")
        # rir_data_path = os.path.join("/mnt", "walkure_public", "tamirs", "pressure_field_orientation_dataset", "robustness_test",name)#,"rir","rir_indoor")
        rir_data_path = os.path.join("/mnt", "walkure_public", "tamirs", "pressure_field_orientation_dataset", "aspect_ratio",name)#,"rir","rir_indoor")
        room_x = round(np.sqrt(1+deformation)*5.0,2)
        room_y = round(5.0/np.sqrt(1+deformation),2)
        margins = 0.02
        minimum_coordinate, maximum_coordinate = get_min_max(room_x, room_y, margins)
        #single_dataset = ModulationDatasetFixedInputSoundFixedGeometryDeformation2d(
        single_dataset = Localization2dGivenOrientationDataset(
            deformation=deformation, data_path=rir_data_path, return_deformation=True, 
            min_coord=minimum_coordinate, max_coord=maximum_coordinate,
            skip_bad_files=True
        )
        datasets.append(single_dataset)
    dataset = torch.utils.data.ConcatDataset(datasets)
    

    '''
    dataset = torch.utils.data.ConcatDataset([
        # rir_aspect_ratio_0.10_5.24_4.77_order_1_0.5_d_0.05
        # /home/gabriele/EARS_project/data/rir/
        ModulationDatasetFixedInputSoundFixedGeometryDeformation2d(deformation=deformation, 
                                                                     rir_data_path=os.path.join("/","home", 
        "gabriele", "EARS_project", "data", "rir", create_name(deformation=deformation),"rir_indoor"),
                                                                     return_deformation=True)
        for deformation in np.arange(0.1, 1.0, 0.05)
    ])
    '''
    # deformation = 0.0
    # name = create_name(deformation=deformation)
    # rir_data_path = os.path.join("/","home", "gabriele", "EARS_project", "data", "rir", create_name(deformation=deformation),"rir_indoor")
    # room_x = round(np.sqrt(1+deformation)*5.0,2)
    # room_y = round(5.0/np.sqrt(1+deformation),2)
    # margins = 0.02
    # minimum_coordinate, maximum_coordinate = get_min_max(room_x, room_y, margins)
    # dataset = ModulationDatasetFixedInputSoundFixedGeometryDeformation2d(
    #     deformation=deformation, rir_data_path=rir_data_path, return_deformation=True, 
    #     min_coord=minimum_coordinate, max_coord=maximum_coordinate
    # )

    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn_absorption_coefficient)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, collate_fn=collate_fn_deformation_orientation,
                                             num_workers=5)

    loss_fn = losses[loss_function]

    # evaluate
    model.eval()
    model = model.to(dev)
    # df will contain a list of tuples (deformation, ground_truth_distance, predicted_distance, loss)
    df = []
    with tqdm(total=len(dataloader)) as pbar:
        pbar.set_description('Evaluating')
        with torch.no_grad():
            for batch in dataloader:
                x, y_cpu, deformation, orientation_cpu = batch
                if x is None or y_cpu is None or deformation is None or orientation_cpu is None:
                    continue
                #print(f'processing {deformation}')
                #x = convolve(x.to(dev), signals=signals)
                x = x.to(dev)
                y = y_cpu.to(dev)
                orientation = orientation_cpu.to(dev)
                y_hat = model(x, orientation=orientation)
                df.extend(
                    list(zip(
                        deformation.cpu().numpy().tolist(), # deformation
                        y_cpu.numpy().tolist(), # ground truth coordinate
                        y_hat.cpu().numpy().tolist(), # predicted coordinate
                        loss_fn(y_hat, y).cpu().numpy().tolist(), # loss
                        orientation_cpu.numpy().tolist() # angle
                    ))
                )
                pbar.update(1)
    
    df = pd.DataFrame(df, columns=['aspect_ratio', 'ground_truth_coordinate', 'predicted_coordinate', 'loss', 'angle'])
    # save dataframe
    # check if file with this name exists
    counter = 0
    name = f'2d_aspect_ratio_robustness_{counter}_{experiment_name}.csv'
    while os.path.exists(name):
        counter += 1
        name = f'2d_aspect_ratio_robustness_{counter}_{experiment_name}.csv'
    df.to_csv(name, index=False)
    print(f"Dataframe saved to {name}")
    return df



def evaluate_shear_deformation_robustness(model: nn.Module, batch_size:int = 10, 
                              loss_function: str = 'rms',
                            dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                            experiment_name:str = None,
                            system_noise_gain:float = 1e-1,
                            picked_shear_angles:list = None) -> pd.DataFrame:
    '''Evaluate the model on the dataset varying the side of the room.

    :arg model: the model to evaluate
    :arg batch_size: the batch size to use when evaluating
    :arg loss_fn: the loss function to use among 'mse' and 'l1'
    :arg dev: the device to use for evaluation
    :returns: a pandas dataframe containing the evaluation metrics
    '''
    # type checking
    assert isinstance(batch_size, int)
    assert loss_function in ['rms', 'l1']
    assert isinstance(dev, torch.device)       
    
    #create_name = lambda deformation: f"rel_units_rir_shear_new_deformation_{deformation}_5.0_5.0_order_1_0.5_d_0.05"
    create_name = lambda deformation: f"s_{int(deformation)}_5.0_5.0_order_1_0.5_d_0.05"
    def load_shear_data(deformation):
        print(f"Loading dataset with deformation {deformation}")
        name = create_name(deformation=deformation)
        rir_data_path = os.path.join("/mnt", "walkure_public", "tamirs", "pressure_field_orientation_dataset", "shear",name) #,"rir","rir_indoor")
        #rir_data_path = os.path.join("/mnt", "walkure_public", "tamirs", "pressure_field_orientation_dataset", "robustness_test",name)
        room_x = 5.0
        room_y = 5.0
        margins = 0.02
        minimum_coordinate, maximum_coordinate = get_min_max(room_x, room_y, margins, angle=deformation)
        single_dataset = Localization2dGivenOrientationDataset(
            deformation=deformation, data_path=rir_data_path, return_deformation=True, 
            min_coord=minimum_coordinate, max_coord=maximum_coordinate,
            shear_angle=deformation
        )
        return single_dataset
    
    datasets = []
    if picked_shear_angles is not None and len(picked_shear_angles)>0:
        deformations = picked_shear_angles
    else: 
        deformations = np.arange(45, 95, 5)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_shear_data, deformation) for deformation in deformations]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                datasets.append(result)
            except Exception as e:
                print(f"Got exception {e}")
                raise e
    dataset = torch.utils.data.ConcatDataset(datasets)

    """
    datasets = []
    for deformation in np.arange(45, 95, 5):
        print(f"Loading dataset with deformation {deformation}")
        name = create_name(deformation=deformation)
        #/mnt/walkure_public/tamirs/rir_robustness_test/shear/
        # rir_data_path = os.path.join("/","mnt", "walkure_public", "tamirs", "rir_robustness_test", "shear", create_name(deformation=deformation),"rir_indoor")
        # rir_data_path = os.path.join("/mnt", "walkure_public", "tamirs", "pressure_field_orientation_dataset", "robustness_test",name) #,"rir","rir_indoor")
        rir_data_path = os.path.join("/mnt", "walkure_public", "tamirs", "pressure_field_orientation_dataset", "shear",name) #,"rir","rir_indoor")
        room_x = 5.0
        room_y = 5.0
        margins = 0.02
        minimum_coordinate, maximum_coordinate = get_min_max(room_x, room_y, margins, angle=deformation)
        #single_dataset = ModulationDatasetFixedInputSoundFixedGeometryDeformation2d(
        single_dataset = Localization2dGivenOrientationDataset(
            deformation=deformation, data_path=rir_data_path, return_deformation=True, 
            min_coord=minimum_coordinate, max_coord=maximum_coordinate,
            shear_angle=deformation
        )
        datasets.append(single_dataset)
    dataset = torch.utils.data.ConcatDataset(datasets)
    """
    
    # deformation=90
    # name = create_name(deformation=deformation)
    # rir_data_path = os.path.join("/","home", "gabriele", "EARS_project", "data", "rir", create_name(deformation=deformation),"rir_indoor")
    # room_x = 5.0
    # room_y = 5.0
    # margins = 0.02
    # minimum_coordinate, maximum_coordinate = get_min_max(room_x, room_y, margins, angle=deformation)
    # dataset = ModulationDatasetFixedInputSoundFixedGeometryDeformation2d(
    #     deformation=deformation, rir_data_path=rir_data_path, return_deformation=True, 
    #     min_coord=minimum_coordinate, max_coord=maximum_coordinate
    # )
    
    
    
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn_absorption_coefficient)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, collate_fn=collate_fn_deformation_orientation,
                                             num_workers=7)

    loss_fn = losses[loss_function]

    # evaluate
    model.eval()
    model = model.to(dev)
    # df will contain a list of tuples (deformation, ground_truth_distance, predicted_distance, loss)
    df = []
    with tqdm(total=len(dataloader)) as pbar:
        pbar.set_description('Evaluating')
        with torch.no_grad():
            for batch in dataloader:
                x, y_cpu, deformation, orientation_cpu = batch
                x = x.to(dev)
                #x = convolve(x.to(dev), signals=signals)
                y = y_cpu.to(dev)
                # apply inverse shear so that the new system of coordinate will be sheared!
                #y = shear(y, -deformation)
                orientation = orientation_cpu.to(dev)
                y_hat = model(x, orientation=orientation)
                loss_value = loss_fn(y_hat, y).cpu().numpy().tolist()
                df.extend(
                    list(zip(
                        deformation.numpy().tolist(), # deformation
                        y_cpu.numpy().tolist(), # ground truth coordinate
                        y_hat.cpu().numpy().tolist(), # predicted coordinate
                        loss_value, # loss
                        orientation_cpu.numpy().tolist() # angle
                    ))
                )
                pbar.update(1)
                pbar.set_postfix({"deformation":deformation[0], "loss_value":loss_value[0]})
    
    df = pd.DataFrame(df, columns=['shear_angle', 'ground_truth_coordinate', 'predicted_coordinate', 'loss', 'angle'])
    # save dataframe
    # check if file with this name exists
    counter = 0
    name = f'2d_shear_deformation_robustness_{counter}_{experiment_name}_deformations_{deformations}.csv'
    while os.path.exists(name):
        counter += 1
        name = f'2d_shear_deformation_robustness_{counter}_{experiment_name}_deformations_{deformations}.csv'
    df.to_csv(name, index=False)
    print(f"Dataframe saved to {name}")
    return df


def evaluate_grid(model: nn.Module, batch_size:int = 10, 
                loss_function: str = 'rms',
                dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                experiment_name:str = None,
                DATA_PATH: str = '/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/default_5.0_5.0_order_1_0.5_d_0.05',
                use_newton_cluster:bool=False,
                use_multi_position:bool=False,
                system_noise_gain:float = 1e-1) -> pd.DataFrame:
    assert isinstance(model, nn.Module)
    assert isinstance(batch_size, int)
    assert loss_function in ['rms', 'l1']

    #DATA_PATH: str = '/mnt/walkure_public/tamirs/pressure_field_2d_no_padding/indoor_recordings_4_rotors_8_mics_d_0.05_mode_indoor_None/'
    
    #dataset = ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2d(DATA_PATH)
    

    if use_multi_position:
        if DATA_PATH is None:
            DATA_PATH = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/shifted_room/default_5.0_5.0_order_1_0.5_d_0.05/"
        compressed_datasets = set(["/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/orig_64_angles/default_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/non_convex_room/non_convex_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/small_dense_room/default_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/shifted_room/default_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/non_convex_room/shifted_non_convex_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/asymmetric_non_convex_room/non_convex_5.0_5.0_order_1_0.5_d_0.05"
                                 ])
        use_hdf5 = args.use_newton_cluster or DATA_PATH in compressed_datasets
        if 'non_convex' in DATA_PATH:
            min_coord: torch.Tensor = torch.tensor([0.93, 0.93])
            max_coord: torch.Tensor = torch.tensor([8.07, 6.07])
        else:
            min_coord: torch.Tensor = torch.tensor([0.93, 0.93])
            max_coord: torch.Tensor = torch.tensor([4.07, 4.07])
        trajectory_fact = trajectory_factory.AllAnglesTrajectoryFactory(use_hdf5=use_hdf5)
        dataset = multi_position_dataset.MultiPositionDataset(DATA_PATH, trajectory_fact, use_newton_cluster=args.use_newton_cluster,
                    min_coord=min_coord, max_coord=max_coord)
        if DATA_PATH == "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/shifted_room/default_5.0_5.0_order_1_0.5_d_0.05/":
            train_split_ratio = 0.8
            val_split_ratio = 0.10
            print(f"Splitting the data in {DATA_PATH} and getting the testing set...")
            _, _, dataset = torch.utils.data.random_split(dataset,
                                                        [int(train_split_ratio * len(dataset)),
                                                        int(val_split_ratio * len(dataset)),
                                                        len(dataset) - int(
                                                            train_split_ratio * len(
                                                                dataset)) - int(
                                                            val_split_ratio * len(dataset))])

    else:
        if DATA_PATH is None:
            DATA_PATH: str = '/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/default_5.0_5.0_order_1_0.5_d_0.05'
        if 'non_convex' in DATA_PATH:
            min_coord: torch.Tensor = torch.tensor([0.93, 0.93])
            max_coord: torch.Tensor = torch.tensor([8.07, 6.07])
        else:
            min_coord: torch.Tensor = torch.tensor([0.93, 0.93])
            max_coord: torch.Tensor = torch.tensor([4.07, 4.07])
        dataset = Localization2dGivenOrientationDataset(DATA_PATH, use_newton_cluster=use_newton_cluster,
                                                        min_coord=min_coord, max_coord=max_coord)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    
    if use_multi_position:
        fnct = multi_position_dataset.collate_fn_orientation
    else:
        fnct = collate_fn_orientation

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=fnct, pin_memory=False,
                                             num_workers=5)

    loss_fn = losses[loss_function]

    model.eval()
    model = model.to(dev)

    df = []
    with tqdm(total=len(dataloader)) as pbar:
        pbar.set_description('Evaluating on a grid')
        with torch.no_grad():
            for x, y_cpu, orientation_cpu in dataloader:
                x, y, orientation = x.to(dev), y_cpu.to(dev), orientation_cpu.to(dev)
                y_hat = model(x, orientation=orientation)
                loss = loss_fn(y_hat, y).cpu().numpy()
                df.extend(
                        list(zip(
                            y_cpu.numpy().tolist(),
                            y_hat.cpu().numpy().tolist(),
                            loss.tolist(),
                            orientation_cpu.numpy().tolist(),
                        ))
                    )
                pbar.update(1)
    df = pd.DataFrame(df, columns=['ground_truth_coordinate', 'predicted_coordinate', 'loss', 'angle'])

    counter = 0
    name = f'grid_{counter}_{experiment_name}_raw.csv'
    while os.path.exists(name):
        counter += 1
        name = f'grid_{counter}_{experiment_name}_raw.csv'

    df.to_csv(name, index=False)
    print(f"Raw dataframe saved to {name}")

    df['ground_truth_coordinate_0'] = df['ground_truth_coordinate'].apply(lambda x: x[0]).astype(float)
    df['ground_truth_coordinate_1'] = df['ground_truth_coordinate'].apply(lambda x: x[1]).astype(float)
    df['predicted_coordinate_0'] = df['predicted_coordinate'].apply(lambda x: x[0]).astype(float)
    df['predicted_coordinate_1'] = df['predicted_coordinate'].apply(lambda x: x[1]).astype(float)
    df['loss_0'] = df['loss'].apply(lambda x: x[0]).astype(float)
    df['loss_1'] = df['loss'].apply(lambda x: x[1]).astype(float)
    df['loss'] = (df['loss_0'] + df['loss_1']) / 2

    df = df.drop(columns=['ground_truth_coordinate', 'predicted_coordinate', 'loss_0', 'loss_1'])

    result_df = df.groupby(['ground_truth_coordinate_0', 'ground_truth_coordinate_1']).apply(
        lambda group: pd.Series({
            'geometric_median': geometric_median(group[['predicted_coordinate_0', 'predicted_coordinate_1']].values)
        })
    ).reset_index()
    final_df = pd.merge(df, result_df, on=['ground_truth_coordinate_0', 'ground_truth_coordinate_1'])
    final_df['geometric_median_loss'] = get_geometric_median_loss(final_df)
    final_df['geometric_median_0'] = final_df['geometric_median'].apply(lambda x: x[0]).astype(float)
    final_df['geometric_median_1'] = final_df['geometric_median'].apply(lambda x: x[1]).astype(float)

    final_df = final_df.drop(columns=['geometric_median', 'angle'])

    output_filename = f'grid_{counter}_{experiment_name}_grid.csv'
    final_df.to_csv(output_filename)
    print(f"Clean dataframe saved to: {output_filename}")

    print(f"Raw mean loss:{final_df.loss.mean()} +- {final_df.loss.mean()}")
    print(f"Geometric median loss: {final_df.geometric_median_loss.mean()} +- {final_df.geometric_median_loss.std()}")

    return df


def evaluate_grid_marked(model: nn.Module, batch_size:int = 10, 
                loss_function: str = 'rms',
                dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                experiment_name:str = None, use_newton_cluster:bool = False,
                system_noise_gain:float = 1e-1) -> pd.DataFrame:
    assert isinstance(model, nn.Module)
    assert isinstance(batch_size, int)
    assert loss_function in ['rms', 'l1']

    #DATA_PATH: str = '/mnt/walkure_public/tamirs/pressure_field_2d_no_padding/indoor_recordings_4_rotors_8_mics_d_0.05_mode_indoor_None/'
    if use_newton_cluster:
        DATA_PATH: str = "/home/gabrieles/EARS/data/pressure_field_orientation_dataset/32_angles/default_5.0_5.0_order_1_0.5_d_0.05/"
    else:
        DATA_PATH: str = '/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/default_5.0_5.0_order_1_0.5_d_0.05'

    DATA_PATH = '/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/orig_64_angles/default_5.0_5.0_order_1_0.5_d_0.05/'
    min_coord:torch.Tensor = torch.tensor([0.93,0.93])
    max_coord:torch.Tensor = torch.tensor([4.07, 4.07])
    #dataset = ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2d(DATA_PATH, min_coord=min_coord, max_coord=max_coord)
    dataset = Localization2dGivenOrientationDataset(DATA_PATH, min_coord=min_coord, max_coord=max_coord, use_newton_cluster=use_newton_cluster)
    # use 80 train, 10% val, 10% test
    train_split_ratio = 0.8
    val_split_ratio = 0.10
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                            [int(train_split_ratio * len(dataset)),
                                                                            int(val_split_ratio * len(dataset)),
                                                                            len(dataset) - int(
                                                                                train_split_ratio * len(
                                                                                    dataset)) - int(
                                                                                val_split_ratio * len(dataset))])

    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_orientation, pin_memory=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_orientation, pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_orientation, pin_memory=False)

    loss_fn = losses[loss_function]

    model.eval()
    model = model.to(dev)

    df = []
    with tqdm(total=len(train_dataloader)) as pbar:
        pbar.set_description('[Train] Evaluating on a grid')
        with torch.no_grad():
            for x, y, orientation in train_dataloader:
                x, y, orientation = x.to(dev), y.to(dev), orientation.to(dev)
                y_hat = model(x, orientation=orientation)
                loss = loss_fn(y_hat, y).cpu().numpy()
                df.extend(
                        list(zip(
                            y.cpu().numpy().tolist(),
                            y_hat.cpu().numpy().tolist(),
                            loss.tolist(),
                            ["training"]*loss.shape[0] 
                        ))
                    )
                pbar.update(1)
    
    with tqdm(total=len(val_dataloader)) as pbar:
        pbar.set_description('[Val] Evaluating on a grid')
        with torch.no_grad():
            for x, y, orientation in val_dataloader:
                x, y, orientation = x.to(dev), y.to(dev), orientation.to(dev)
                y_hat = model(x, orientation=orientation)
                loss = loss_fn(y_hat, y).cpu().numpy()
                df.extend(
                        list(zip(
                            y.cpu().numpy().tolist(),
                            y_hat.cpu().numpy().tolist(),
                            loss.tolist(),
                            ["validation"]*loss.shape[0] 
                        ))
                    )
                pbar.update(1)
    
    with tqdm(total=len(test_dataloader)) as pbar:
        pbar.set_description('[Test] Evaluating on a grid')
        with torch.no_grad():
            for x, y, orientation in test_dataloader:
                x, y, orientation = x.to(dev), y.to(dev), orientation.to(dev)
                y_hat = model(x, orientation=orientation)
                loss = loss_fn(y_hat, y).cpu().numpy()
                df.extend(
                        list(zip(
                            y.cpu().numpy().tolist(),
                            y_hat.cpu().numpy().tolist(),
                            loss.tolist(),
                            ["testing"]*loss.shape[0] 
                        ))
                    )
                pbar.update(1)


    df = pd.DataFrame(df, columns=['ground_truth_coordinate', 'predicted_coordinate', 'loss', 'dataset'])

    counter = 0
    name = f'grid_marked_{counter}_{experiment_name}.csv'
    while os.path.exists(name):
        counter += 1
        name = f'grid_marked_{counter}_{experiment_name}.csv'

    df.to_csv(name, index=False)
    print(f"Dataframe saved to {name}")

    return df


def evaluate_phase_system_noise_robustness(model: nn.Module, batch_size:int = 10, 
                              loss_function: str = 'rms',
                            dev: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                            experiment_name:str = None,
                            use_newton_cluster:bool = False,
                            DATA_PATH:str = None, 
                            picked_noise_intensities:list = None) -> pd.DataFrame:
    '''Evaluate the model on the dataset varying the system phase noise.

    :arg model: the model to evaluate
    :arg batch_size: the batch size to use when evaluating
    :arg loss_fn: the loss function to use among 'mse' and 'l1'
    :arg dev: the device to use for evaluation
    :returns: a pandas dataframe containing the evaluation metrics
    '''
    # type checking
    assert isinstance(batch_size, int)
    assert loss_function in ['rms', 'l1']
    assert isinstance(dev, torch.device)

    if DATA_PATH is None:
        DATA_PATH = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/shifted_room/default_5.0_5.0_order_1_0.5_d_0.05/"
    dataset = Localization2dGivenOrientationDataset(DATA_PATH, use_newton_cluster=use_newton_cluster)

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_orientation, pin_memory=True,
                                                  num_workers=5)

    loss_fn = losses[loss_function]

    # evaluate
    model.eval()
    model = model.to(dev)
    # df will contain a list of tuples (standard_deviation, ground_truth_distance, predicted_distance, loss, angle)
    df = []
    n_experiments = 3
    if picked_noise_intensities is not None and len(picked_noise_intensities)> 0:
        noise_intensities = picked_noise_intensities
    else:
        noise_intensities = np.arange(0,-18,-3).tolist()
    print(f"checking {noise_intensities} noise levels...")
    
    for noise_intensity in noise_intensities:
        print(f"Processing noise intensity: {noise_intensity}")
        with tqdm(total=len(test_dataloader)) as pbar:
            pbar.set_description('Evaluating')
            with torch.no_grad():
                for batch in test_dataloader:
                    x, y_cpu, orientation_cpu = batch
                    x = x.to(dev)
                    y = y_cpu.to(dev)
                    orientation = orientation_cpu.to(dev)
                    avg_loss = 0
                    avg_predicted_distance = 0
                    for i in range(n_experiments):
                        if noise_intensity == 'inf':
                            y_hat = model(x, orientation=orientation)
                        else:
                            y_hat = model(x, orientation=orientation, system_noise_snr_db=noise_intensity)
                        loss = loss_fn(y_hat, y).cpu().numpy()
                        avg_loss = (i*avg_loss+loss)/(i+1)
                        avg_predicted_distance = (i*avg_predicted_distance+y_hat)/(i+1)

                    df.extend(
                        list(zip(
                            [noise_intensity]*avg_loss.shape[0], # intensity of the noise
                            y_cpu.numpy().tolist(), # ground truth coordinate
                            avg_predicted_distance.cpu().numpy().tolist(), # predicted coordinate
                            avg_loss.tolist(), # loss
                            [False]*avg_loss.shape[0], # False signals that we are working with the validation dataset
                            orientation_cpu.numpy().tolist(),
                        ))
                    )
                    pbar.update(1)
    
    df = pd.DataFrame(df, columns=['snr','ground_truth_coordinate', 'predicted_coordinate', 'loss', 'is_training_dataset', 'orientation'])
    counter = 0
    if experiment_name is not None:
        name = f'phase_system_noise_robustness_{counter}_{experiment_name}_noise_intensities_{noise_intensities}.csv'
    else:
        name = f'phase_system_noise_robustness_{counter}_noise_intensities_{noise_intensities}.csv'
    while os.path.exists(name):
        counter += 1
        if experiment_name is not None:
            name = f'phase_system_noise_robustness_{counter}_{experiment_name}_noise_intensities_{noise_intensities}.csv'
        else:
            name = f'phase_system_noise_robustness_{counter}_noise_intensities_{noise_intensities}.csv'
    raw_name = f"{name.removesuffix('.csv')}_raw.csv"
    df.to_csv(raw_name, index=False)
    print(f"Raw dataframe saved to {raw_name}")
    
    df['ground_truth_coordinate_0'] = df['ground_truth_coordinate'].apply(lambda x: x[0]).astype(float)
    df['ground_truth_coordinate_1'] = df['ground_truth_coordinate'].apply(lambda x: x[1]).astype(float)
    df['predicted_coordinate_0'] = df['predicted_coordinate'].apply(lambda x: x[0]).astype(float)
    df['predicted_coordinate_1'] = df['predicted_coordinate'].apply(lambda x: x[1]).astype(float)
    df['loss_0'] = df['loss'].apply(lambda x: x[0]).astype(float)
    df['loss_1'] = df['loss'].apply(lambda x: x[1]).astype(float)
    df['loss'] = (df['loss_0'] + df['loss_1']) / 2

    df = df.drop(columns=['ground_truth_coordinate', 'predicted_coordinate', 'loss_0', 'loss_1'])

    df = df.drop(columns='is_training_dataset')
    # save dataframe
    # check if file with this name exists
    # df.to_csv(name, index=False)
    # print(f"Dataframe saved to {name}")
    result_df = df.groupby(['ground_truth_coordinate_0', 'ground_truth_coordinate_1']).apply(
        lambda group: pd.Series({
            'geometric_median': geometric_median(group[['predicted_coordinate_0', 'predicted_coordinate_1']].values)
        })
    ).reset_index()
    final_df = pd.merge(df, result_df, on=['ground_truth_coordinate_0', 'ground_truth_coordinate_1'])
    final_df['geometric_median_loss'] = get_geometric_median_loss(final_df)
    final_df['geometric_median_0'] = final_df['geometric_median'].apply(lambda x: x[0]).astype(float)
    final_df['geometric_median_1'] = final_df['geometric_median'].apply(lambda x: x[1]).astype(float)

    final_df = final_df.drop(columns=['geometric_median', 'orientation'])
    
    df_summary_mean = df.groupby('snr')['loss'].mean()
    df_summary_std = df.groupby('snr')['loss'].std()
    df_summary_geometric_median_mean = df.groupby('snr')['geometric_median_loss'].mean()
    df_summary_geometric_median_std = df.groupby('snr')['geometric_median_loss'].std()
    # df_summary = pd.merge(df_summary_mean, df_summary_std, on='snr')
    df_summary = df_summary_mean.merge(df_summary_std, on='snr').\
        merge(df_summary_geometric_median_mean, on='snr').\
        merge(df_summary_geometric_median_std, on='snr')
    #df_summary.to_csv(f'noise_in_input_robustness_0_summary.csv')
    #df_summary.rename(columns={'loss_x':'mean', 'loss_y':'std'}, inplace=True)
    df_summary.rename(columns={
        'geometric_median_loss_x':'geometric_median_loss_mean', 
        'geometric_median_loss_y':'geometric_median_loss_std',
        'mean': 'loss_mean',
        'std': 'loss_std'}, inplace=True)

    if experiment_name is not None:
        summary_name = f'phase_system_noise_robustness_{counter}_{experiment_name}_summary_noise_intensities_{noise_intensities}.csv'
    else:
        summary_name = f'phase_system_noise_robustness_{counter}_summary_noise_intensities_{noise_intensities}.csv'
    df_summary.to_csv(summary_name)
    print(f"Summary dataframe saved to {summary_name}")
    return df_summary

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default=None, help='Name of the experiment')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--loss_function', type=str, default='rms', help='Loss function')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation')
    parser.add_argument('--robustness_test', choices=['absorption', 'input_noise', 'angle', 'aspect_ratio', 'uniform', 'shear', 'grid', 'grid_marked', 'system_phase_noise'], type=str, help='Robustness test to perform')
    parser.add_argument('--use-newton-cluster', default=False, action='store_true', help='Whether we are using the Newton cluster')
    parser.add_argument('--signal', choices=['original', 'december'], default='december', help='Choose which signal to convolve the RIRs with')
    parser.add_argument('--data-path', default=None, type=str, help='path to the data to load (to be used in grid test)')
    parser.add_argument('--use-multi-position', action='store_true', default=False, help='Whether to aggregate the prediction from multiple angles')

    parser.add_argument('--picked-noise-intensities', default=None, type=float, nargs="*", help="Choose the specific noise intensities to test (used for the input_noise test)")
    parser.add_argument('--testing-absorption-coefficients', default=None, type=float, nargs="*", help="Choose the specific absorption coefficients to test (used for the absorption test)")
    parser.add_argument('--picked-shear-angles', default=None, type=float, nargs="*", help="Choose the specific shear angles to test (used for the shear test)")
    parser.add_argument('--picked-system-noise-intensities', default=None, type=float, nargs="*", help="Choose the specific system phase noise intensities to test (used for the system_phase_noise test)")


    # parser.add_argument('--dataset', type=str, default='rir', help='Dataset to use for robustness test')
    # parser.add_argument('--dataset_path', type=str, default=None, help='Path to the dataset')
    # parser.add_argument('--n_experiments', type=int, default=3, help='Number of experiments to perform for each datapoint')
    # parser.add_argument('--noise_intensities', type=str, default='5,10,15,20,25,30,35,40,45,50,55,60,65,70,75', help='Noise intensities to use for the robustness test')
    # parser.add_argument('--absorption_coefficients', type=str, default='0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4', help='Absorption coefficients to use for the robustness test')
    # parser.add_argument('--standard_deviations', type=str, default='1e-4,1e-3,1e-2,1e-1', help='Standard deviations to use for the robustness test')
    # parser.add_argument('--n_workers', type=int, default=0, help='Number of workers to use for the dataloader')
    # parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to use for the robustness test')
    # parser.add_argument('--n_samples_per_absorption_coefficient', type=int, default=None, help='Number of samples to use for the robustness test')
    # parser.add_argument('--n_samples_per_noise_intensity', type=int, default=None, help='Number of samples to use for the robustness test')
    # parser.add_argument('--n_samples_per_standard_deviation', type=int, default=None, help='Number of samples to use for the robustness test')
    # parser.add_argument('--n_samples_per_experiment', type=int, default=None, help='Number of samples to use for the robustness test')
    # parser.add_argument('--n_samples_per_dataset', type=int, default=None, help='Number of samples to use for the robustness test')
    # parser.add_argument('--n_samples_per_dataset_per_experiment', type=int, default=None, help='Number of samples to use for the robustness test')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # model = Localization_Model(hidden_dim=1024, num_layers=3, num_heads=1,num_microphones=8,
    #                            dropout=0.0,decimation_rate=10, res=320, learn_shift=False,
    #                            initialization='constant', n_rotors=4,
    #                            interp_gap=2, rec_len=1501)
    
    # ATTENTION: interp_gap must always be set to 2 as it's the last value of interpolation gap set during training!
    # model = Localization_Model(hidden_dim=1024, num_layers=3, num_heads=1,num_microphones=8,
    #                            dropout=0.0,decimation_rate=10, res=320, learn_shift=False,
    #                            initialization='zero', n_rotors=4,
    #                            interp_gap=2, rec_len=1501, num_coordinates=2)
    
    # model = Localization_Model(hidden_dim=1024, num_layers=3, num_heads=1,num_microphones=8,
    #                            dropout=0.0,decimation_rate=10, res=320, learn_shift=False,
    #                            initialization='zero', n_rotors=4,
    #                            interp_gap=2, rec_len=1501, num_coordinates=2,
    #                            use_wide_cnn=True)
    
    # model = Localization_Model(hidden_dim=1024, num_layers=3, num_heads=1,num_microphones=8,
    #                            dropout=0.0,decimation_rate=10, res=320, learn_shift=False,
    #                            initialization='zero', n_rotors=4,
    #                            interp_gap=2, rec_len=1501, num_coordinates=2)

    # model = Localization_Model(hidden_dim=1024, num_layers=3, num_heads=1,num_microphones=8,
    #                            dropout=0.0,decimation_rate=10, res=320, learn_shift=False,
    #                            initialization='zero', n_rotors=4,
    #                            interp_gap=2, rec_len=1501, num_coordinates=2,
    #                            max_vel=None, max_acc=None, project=False)

    # model = Localization_Model(hidden_dim=1024, num_layers=3, num_heads=1,num_microphones=8,
    #                           dropout=0.0,decimation_rate=10, res=320, learn_shift=False,
    #                           initialization='zero', n_rotors=4,
    #                          interp_gap=2, rec_len=1501, num_coordinates=2,
    #                           max_vel=None, max_acc=None, project=False,
    #                           use_fourier=False, basis_size=100, cutoff=10)
    
    args = parse_arguments()

    #path = "/home/tamir.shor/EARS/freeze_epoch_5_low_sublr_multisine/summary/test/best_model.pt"
    #path = "/home/gabrieles/ears/code/summary/inject_noise/best_model.pt"
    #path = "/home/gabrieles/ears/code/summary/joint_slow/best_model.pt"
    #path = "/home/gabrieles/ears/code/summary/separate/best_model.pt"
    if args.model_path is None:
        raise ValueError("The path to the model has not been passed. Please set --model-path to the right path!")
        path = "/home/gabrieles/ears/code/summary/constant_phase/best_model.pt"
    else:
        path = args.model_path
    
    assert args.experiment_name is not None, "Please indicate a name to attribute to the experiment by setting --experiment_name"

    with open(os.path.join(os.path.dirname(path), 'args.txt')) as f:
        arguments = Munch(**eval(f.read()))

    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    RECORDING_LENGTH: int = 1_501
    if arguments.use_2d or arguments.use_2d_given_orientation:
        num_coordinates = 2
    elif arguments.use_orientation:
        num_coordinates = 3
    else:
        num_coordinates = 1

    model = Localization_Model(
        hidden_dim=arguments.hidden_dim,
        num_layers=arguments.num_layers,
        num_heads=arguments.num_heads,
        num_microphones=arguments.num_mics,
        dropout=arguments.drop_prob,
        decimation_rate=arguments.decimation_rate,
        max_vel=arguments.v_max,
        max_acc=arguments.a_max,
        res=arguments.resolution,
        learn_shift=arguments.no_phase_shift_learn,
        initialization=arguments.initialization,
        n_rotors=arguments.num_rotors,
        interp_gap=arguments.interp_gap,
        rec_len=RECORDING_LENGTH,
        projection_iters=arguments.proj_iters,
        project=arguments.project,
        device=arguments.device,
        no_fourier=arguments.no_fourier,
        inject_noise=arguments.inject_noise,
        standardize=arguments.standardize,
        num_coordinates=num_coordinates,
        use_cnn=arguments.use_cnn,
        use_wide_cnn=arguments.use_wide_cnn,
        use_fourier=arguments.use_fourier,
        cutoff=arguments.fourier_cutoff_freq,
        basis_size=arguments.fourier_basis_size,
        use_cosine=arguments.use_cosine_basis,
        high_resolution_stft=arguments.high_resolution_stft,
        sampling_interpolation_method = arguments.sampling_interpolation_method,
        phase_modulation_snr_db_list = arguments.phase_modulation_snr_db_list,
        init_size = arguments.init_size,
        use_aggregate=False if "use_aggregate" not in arguments else arguments.use_aggregate,
        use_sound_transformer=False if "use_sound_transformer" not in arguments else arguments.use_sound_transformer,
        use_orientation_aggregate=False if "use_2d_given_orientation" not in arguments else arguments.use_2d_given_orientation,
        use_orientation_transformer=False if "use_orientation_transformer" not in arguments else arguments.use_orientation_transformer,
        simulate_system=False if "simulate_system" not in arguments else arguments.simulate_system,
        use_orientation_aggregate_mlp=False if "use_orientation_aggregate_mlp" not in arguments else arguments.use_orientation_aggregate_mlp,
        use_orientation_aggregate_attention=False if "use_orientation_aggregate_attention" not in arguments else arguments.use_orientation_aggregate_attention,
        simulate_system_method="fourier" if "simulate_system_method" not in arguments else arguments.simulate_system_method,
    ).to(arguments.device).double()
    #model.phase_model.interp_gap = int(32 * args.gap_factor)

    if args.use_multi_position:
        aggregator_model = aggregator.GeometricMedianAggregator()
        model = master.ParallelMultiPositionModel(single_position_model=model, aggregator=aggregator_model)
        model = model.to(arguments.device).double()
        model.eval()
        model = model.double()
        
    parameters = torch.load(path)
    model.load_state_dict(parameters['model'])
    model.eval()
    model = model.double()


    if not args.use_newton_cluster:
        if args.signal == 'original':
            signals_path = '/mnt/walkure_public/tamirs/signals.npy'
        elif args.signal == 'december':
            signals_path = '/mnt/walkure_public/tamirs/correct_december_signals.npy'
        else:
            raise ValueError("Please choose a value of signal between original and december")
        signals = torch.from_numpy(np.load(signals_path)).unsqueeze(0)
        #signals = torch.from_numpy(np.load('/mnt/walkure_public/tamirs/december_signals.npy')).unsqueeze(0)
        signals = signals.to(args.device, dtype=torch.float64)

    if args.robustness_test == 'absorption':
        print("Running absorption coefficient test...")
        df = evaluate_absorption_coefficient_robustness_2d(model, experiment_name=args.experiment_name,
                                                           batch_size=args.batch_size,
                                                           loss_function=args.loss_function,
                                                           dev=torch.device(args.device),
                    testing_absorption_coefficients=args.testing_absorption_coefficients)
        print('absorption coeffient done')
    elif args.robustness_test == 'input_noise':
        print("Running input noise robustness test...")
        df = evaluate_input_noise_robustness(model, experiment_name=args.experiment_name, 
                                            batch_size=args.batch_size, 
                                            loss_function=args.loss_function, 
                                            dev=torch.device(args.device),
                                            use_newton_cluster=args.use_newton_cluster,
                system_noise_gain=1e-1 if not "system_noise_gain" in arguments else arguments.system_noise_gain,
                picked_noise_intensities=args.picked_noise_intensities)
        print('noise done')
    elif args.robustness_test == 'angle':
        df = evaluate_angle_deformation_robustness(model, batch_size=args.batch_size,
                                                experiment_name=args.experiment_name)
    elif args.robustness_test == 'aspect_ratio':
        print("Running aspect ratio test...")
        df = evaluate_aspect_ratio_deformation_robustness(model, batch_size=args.batch_size,
                                                          experiment_name=args.experiment_name,
                system_noise_gain=1e-1 if not "system_noise_gain" in arguments else arguments.system_noise_gain)
        print('aspect ratio done')
    elif args.robustness_test == 'uniform':
        print("Running uniform test...")
        df = evaluate_uniform_deformation_robustness(model, batch_size=args.batch_size,
                                                     experiment_name=args.experiment_name,
                system_noise_gain=1e-1 if not "system_noise_gain" in arguments else arguments.system_noise_gain )
        print('uniform done')
    elif args.robustness_test == 'shear':
        print("Running shear deformation test...")
        df = evaluate_shear_deformation_robustness(model, batch_size=args.batch_size,
                                                   experiment_name=args.experiment_name,
                system_noise_gain=1e-1 if not "system_noise_gain" in arguments else arguments.system_noise_gain,
                picked_shear_angles=args.picked_shear_angles)
        print('shear done')
    elif args.robustness_test == 'grid':
        print("Evaluating the model on a grid...")
        df = evaluate_grid(model, batch_size=args.batch_size, loss_function=args.loss_function,
                           dev=args.device, experiment_name=args.experiment_name,
                           DATA_PATH=args.data_path, use_newton_cluster=args.use_newton_cluster,
                           use_multi_position=args.use_multi_position,
                system_noise_gain=1e-1 if not "system_noise_gain" in arguments else arguments.system_noise_gain)
        print("Evaluation on a grid done")
    elif args.robustness_test == 'grid_marked':
        print("Evaluating the model on a grid marking which point is in the training set and which is not...")
        df = evaluate_grid_marked(model, batch_size=args.batch_size, loss_function=args.loss_function,
                           dev=args.device, experiment_name=args.experiment_name,
                           use_newton_cluster=args.use_newton_cluster,
                system_noise_gain=1e-1 if not "system_noise_gain" in arguments else arguments.system_noise_gain)
        print("Evaluation on a grid (marked) done")
    elif args.robustness_test == 'system_phase_noise':
        print("Running system phase noise robustness test...")
        df = evaluate_phase_system_noise_robustness(model, experiment_name=args.experiment_name, 
                                            batch_size=args.batch_size, 
                                            loss_function=args.loss_function, 
                                            dev=torch.device(args.device),
                                            use_newton_cluster=args.use_newton_cluster,
                picked_noise_intensities=args.picked_system_noise_intensities)
        print("System phase noise test done")
    else:
        raise ValueError("Please specify under --robustness_test which test you want to perform")
    print(df)

