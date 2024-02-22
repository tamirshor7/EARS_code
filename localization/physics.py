'''
This file contains all of the functions that are used to compute physical quantities and to between different units.
It also includes the constants that have been used in our experiments.
'''
# Imports
import numpy as np
import torch
from typing import Union
from scipy.signal import convolve
from scipy.signal.windows import gaussian

# Constants
# FS (sampling frequency) is the number of samples per second [Hz = SAMPLES/s]
FS: float = 3003.735603735763
# RPS (rotations per second) is the number of rotations per second [Hz = ROTATIONS/s]
RPS: float = 23.46668440418565
# SAMPLES_PER_ROTATION is the number of samples per rotation (a.k.a. revolutions) (FS / RPS) [SAMPLES/ROTATION]
SAMPLES_PER_ROTATION: int = 128
# DT (delta time) is the time between samples [s/SAMPLE]
DT: float = (1/FS)/0.34 #(1024 for numerical stability unit conversion)
PLOT_DT: float = 1/FS
# ALLOWED_WINDOW_TYPES are the allowed types of windows that convolve the velocity of the phase to get the integrated velocity
ALLOWED_WINDOW_TYPES:set = set(['ones', 'gaussian'])

class Physics:
    '''
    This class contains all of the functions that are used to compute physical quantities and to between different units.
    It also includes the constants that have been used in our experiments.
    '''
    def __init__(self, recording_length=1024, fs=FS, rps=RPS, samples_per_rotation=SAMPLES_PER_ROTATION, 
                 #dt=DT, 
                 dt=PLOT_DT,
                 plot_dt = PLOT_DT, allowed_window_types=ALLOWED_WINDOW_TYPES) -> None:
        # recording_length is in number of samples!
        self.recording_length = recording_length
        self.fs = fs
        self.rps = rps
        self.samples_per_rotation = samples_per_rotation
        self.dt = dt
        self.plot_dt = plot_dt
        self.allowed_window_types = allowed_window_types
        self.number_of_revolutions = self.recording_length/self.samples_per_rotation

    # Functions
    def number_of_raw_samples_to_number_of_revolutions_range(self, number_of_raw_samples: float) -> float:
        '''
        This function converts the number of raw samples to the number of revolutions
        that would have occurred in that time.
        It is usually used to get the x-axis of a plot in revolutions.
        '''
        return np.arange(number_of_raw_samples)/ self.samples_per_rotation

    def number_of_revolutions_to_number_of_raw_samples(self, number_of_revolutions: float) -> float:
        '''
        This function converts number of revolutions to number_of_raw_samples.
        It is usually used to get the x-axis of a plot in raw samples.
        '''
        #return number_of_revolutions * self.samples_per_rotation/self.number_of_revolutions #over 8 due to stability scaling
        return number_of_revolutions * self.samples_per_rotation

    def number_of_raw_samples_to_seconds_range(self, number_of_raw_samples: float) -> np.ndarray:
        '''
        This function takes as input the length of a signal in raw samples and returns an array of the same length
        that contains the time in seconds at each sample.
        It is usually used to get the x-axis of a plot in seconds.
        '''
        return np.arange(number_of_raw_samples) * self.dt

    def seconds_to_raw_samples(self,second: float) -> int:
        return second/self.dt

    def raw_samples_to_seconds(self, raw_sample:int) -> float:
        return raw_sample*self.dt

    def raw_samples_to_revolutions(self, sample: int) -> float:
        return sample/self.samples_per_rotation

    def radians_per_second_to_degrees_per_revolution(self,raw_samples: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        '''
        This function takes as input the samples in radians per second and returns the corresponding angle in degrees per revolution.
        '''
        if isinstance(raw_samples, torch.Tensor):
            return torch.rad2deg(raw_samples) / self.rps
        if isinstance(raw_samples, np.ndarray):
            return np.rad2deg(raw_samples) / self.rps
        raise TypeError(f'raw_samples must be a torch.Tensor or a np.ndarray, not {type(raw_samples)}')

    def number_of_revolutions_to_number_of_raw_samples(self, number_of_revolutions: float) -> float:
        '''
        This function converts number of revolutions to number_of_raw_samples.
        It is used in get_integrated_velocity.
        '''
        #return number_of_revolutions * self.samples_per_rotation/self.number_of_revolutions #over 8 due to stability scaling
        return number_of_revolutions * self.samples_per_rotation

    def get_derivative(self, x: Union[np.ndarray, torch.Tensor], dt = DT) -> Union[np.ndarray, torch.Tensor]:
        '''
        This function returns the finite difference approximation of the first derivative.
        '''
        if isinstance(x, torch.Tensor):
            return torch.diff(x, n = 1, dim = -1) / self.dt
        if isinstance(x, np.ndarray):
            return np.diff(x, n = 1, axis = -1) / self.dt
        raise TypeError(f'x must be a torch.Tensor or a np.ndarray, not {type(x)}')

    def _gaussian_window_numpy(self, fwhm:Union[int, float], use_degrees_per_revolution = True):
        if use_degrees_per_revolution:
            window_size = self.number_of_revolutions_to_number_of_raw_samples(fwhm)
        else:
            window_size = fwhm

        std = fwhm/(2*np.sqrt(2*np.log(2)))
        filter = gaussian(M=window_size, std=std)
        return filter

    def _gaussian_window_torch(self, fwhm:Union[int, float], dtype, device, use_degrees_per_revolution = True):
        # TODO:
        # 1. make sure that the Gaussian curve is scaled in the right way so that it is in the Fourier domain
        # 2. Truncate it to 3 sigma
        # 3. grab the amplitudes, not the signal itself! (Move the code somewhere else)
        if use_degrees_per_revolution:
            window_size = self.number_of_revolutions_to_number_of_raw_samples(fwhm)
        else:
            window_size = fwhm

        std = fwhm/(2*torch.sqrt(2*torch.log(2)))
        filter = torch.signal.windows.gaussian(M=window_size, std=std, dtype=dtype, device=device)
        return filter

    def get_integrated_velocity(self, x: Union[np.ndarray, torch.Tensor], window_size:Union[int, float, list], window_type:str='ones', use_degrees_per_revolution = True) -> Union[np.ndarray, torch.Tensor]:
        '''
        This function returns a windowed sum of the finite difference approximation of the first derivative.
        window_size: if window_type = 'ones' then is the number of samples to sum over (if use_degrees_per_revolution is True then it is measured in number of revolutions, otherwise it is in number of samples).
                    if window type = 'gaussian' then it is a list containing the fwhm values  
        '''
        #assert window_type in ALLOWED_WINDOW_TYPES, f"You chose a window_type of {window_type} whereas the allowed window types are {ALLOWED_WINDOW_TYPES}. Please change the window_type argument"
        assert not (isinstance(window_size, float) and (not use_degrees_per_revolution)), "If you set use_degrees_per_revolution to False (hence you are expressing the window_size in number of samples) you cannot set window_size to a float"

        if isinstance(x, torch.Tensor):
            return self._get_integrated_velocity_torch(x, window_size, window_type=window_type, use_degrees_per_revolution = use_degrees_per_revolution, dt = self.dt)
        if isinstance(x, np.ndarray):
            return self._get_integrated_velocity_numpy(x, window_size, window_type=window_type, use_degrees_per_revolution = use_degrees_per_revolution, dt = self.dt)     
        raise TypeError(f'x must be a torch.Tensor or a np.ndarray, not {type(x)}')

    def _get_integrated_velocity_numpy(self, x: np.ndarray, window_size, window_type, use_degrees_per_revolution = True) -> np.ndarray:
        velocity = self.get_derivative(x, dt = self.dt)
        if use_degrees_per_revolution:
            velocity = self.radians_per_second_to_degrees_per_revolution(velocity)
            window_size = self.number_of_revolutions_to_number_of_raw_samples(window_size)
        
        if window_type == 'ones':
            kernel = np.ones(window_size)
        elif window_type == 'gaussian':
            kernel = self._gaussian_window_numpy(window_size, use_degrees_per_revolution=use_degrees_per_revolution)
        kernel = np.expand_dims(kernel, axis=0)
        kernel = kernel/np.sum(kernel, axis=-1, keepdims=True)
        return convolve(velocity, kernel, 'valid')

    def _get_integrated_velocity_torch(self, x: torch.Tensor, window_size, window_type, use_degrees_per_revolution = True) -> torch.Tensor:
        velocity = self.get_derivative(x, dt = self.dt)
        if use_degrees_per_revolution:
            velocity = self.radians_per_second_to_degrees_per_revolution(velocity)
            window_size = round(self.number_of_revolutions_to_number_of_raw_samples(window_size))

        if len(velocity.shape) in [2,3]:
            kernel_shape = [velocity.shape[-2],1,window_size]
            group_size = velocity.shape[-2]
            axis_to_add = 3-len(velocity.shape)
            velocity = torch.reshape(velocity, [1]*axis_to_add + list(velocity.shape))
        else:
            kernel_shape = [1,1,window_size]
            group_size = 1 
            axis_to_add = 2
            velocity = torch.reshape(velocity, [1]*axis_to_add + list(velocity.shape))
        '''
        if window_type == 'ones':
            kernel = torch.ones(*kernel_shape, device=x.device, dtype=torch.double)
        elif window_type == 'gaussian':
            kernel = _gaussian_window_torch(window_size, dtype=torch.double, device=x.device, use_degrees_per_revolution=use_degrees_per_revolution)
            kernel = torch.reshape(kernel, *kernel_shape)
        '''
        kernel = torch.ones(*kernel_shape, device=x.device, dtype=torch.double)

        kernel = kernel/kernel.sum(dim=-1, keepdim=True)
        # USE CIRCULAR PADDING
        velocity = torch.nn.functional.pad(velocity, (window_size//2,window_size//2), mode='circular').double()
        return torch.nn.functional.conv1d(velocity, kernel, padding = 0, groups=group_size).squeeze()