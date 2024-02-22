import torch
import numpy as np
import os
import argparse
from EARS.forward_model.forward_model import main_forward, add_args
from typing import Union

@torch.jit.script
def _batch_convolve_with_rir(sound: torch.Tensor, rir: torch.Tensor):
    '''Convolve the sound with the rir'''
    # rir has shape   (num_rirs, num_mics, num_sources, max_rir_length) ~ (10,8,1024,3000)
    # sound has shape (num_sounds, num_sources, max_sound_length) ~ (1,1024, 3000)
    # we want to convolve each sound with all the rirs
    # output should have shape (num_sounds, num_rirs, num_mics, max_rir_length + max_sound_length - 1)

    num_sounds, num_sources, _ = sound.shape
    num_rirs, num_mics, _, max_rir_length = rir.shape

    # More expanded implementation of the last line (advantage: more readable; disadvantage: uses temporary variables which might increase memory consumption):

    # rir = rir.view(num_rirs * num_mics, num_sources, max_rir_length).flip([2])  # shape: (num_rirs * num_mics, num_sources, max_rir_length)
    # #rir = torch.flip(rir, dims=[2])  # shape: (num_rirs * num_mics, num_sources, max_rir_length)
    # output = torch.nn.functional.conv1d(sound, rir, padding=max_rir_length - 1)  # shape: (num_sounds, num_rirs * num_mics, max_sound_length + max_rir_length - 1)
    # output = output.view(num_sounds, num_rirs, num_mics, -1)  # shape: (num_sounds, num_rirs, num_mics, max_sound_length + max_rir_length - 1)
    # return output

    # The following code is equivalent to the above code, but it is faster
    return torch.nn.functional.conv1d(sound,  rir.view(num_rirs * num_mics, num_sources, max_rir_length).flip([2]), padding=max_rir_length - 1).view(num_sounds, num_rirs, num_mics, -1)

@torch.jit.script
def _chunked_batch_convolve_with_rir(sound: torch.Tensor, rir: torch.Tensor):
    '''
    Convolve the sound with the rir
    This implementation does NOT compute convolution in the Fourier domain.
    It's the most chunked version of convolution, hence it is slower but it uses less memory.
    '''
    # rir has shape   (num_rirs, num_mics, num_sources, max_rir_length) ~ (10,8,1024,3000)
    # sound has shape (num_sounds, num_sources, max_sound_length) ~ (1,1024, 3000)
    # we want to convolve each sound with all the rirs
    # output has shape (num_sounds, num_rirs, num_mics, max_rir_length + max_sound_length - 1)

    num_sounds, num_sources, _ = sound.shape
    num_rirs, num_mics, _, max_rir_length = rir.shape

    # The more we chunk the higher the run-time and the lower the memory complexity
    output = torch.zeros((num_sounds, num_rirs, num_mics, max_rir_length + sound.shape[-1] - 1), device=sound.device, dtype=sound.dtype)
    for i in range(num_sounds):
        for j in range(num_rirs):
            for k in range(num_mics):
                for l in range(num_sources):
                    output[i, j, k] += torch.nn.functional.conv1d(sound[i, l].unsqueeze(0), rir[j, k, l].unsqueeze(0).unsqueeze(0).flip([-1]), padding=max_rir_length - 1).squeeze(0)
                # uncomment and cancel the for loop over num_sources to chunk less
                #output[i, j, k] = torch.nn.functional.conv1d(sound[i], rir[j, k].unsqueeze(0).flip([-1]), padding=max_rir_length - 1).squeeze(0)
            # uncomment and cancel the for loop over num_mics to chunk less
            #output[i, j] = torch.nn.functional.conv1d(sound[i], rir[j].flip([-1]), padding=max_rir_length - 1)
    return output

@torch.jit.script
def _fourier_batch_convolve_with_rir(sound: torch.Tensor, rir: torch.Tensor):
    '''Convolve the sound with the rir using the fourier transform'''
    # rir has shape   (num_rirs, num_mics, num_sources, max_rir_length)
    # sound has shape (num_sounds, num_sources, max_sound_length)
    # we want to convolve each sound with all the rirs
    # output should have shape (num_sounds, num_rirs, num_mics, max_rir_length + max_sound_length - 1)

    num_sounds, num_sources, _ = sound.shape
    num_rirs, num_mics, _, max_rir_length = rir.shape
    output_time_shape = max_rir_length + sound.shape[-1] - 1

    rir = rir.view(num_rirs * num_mics, num_sources, max_rir_length)
    rir_fourier = torch.fft.rfft(rir, dim=-1, n=output_time_shape)
    sound_fourier = torch.fft.rfft(sound, dim=-1, n=output_time_shape)
    output_fourier = sound_fourier.unsqueeze(1)* rir_fourier.unsqueeze(0)
    output = torch.fft.irfft(output_fourier, n=output_time_shape, dim=-1)
    return output.view(num_sounds, num_rirs, num_mics, -1)[:,:,:,:output_time_shape]

@torch.jit.script
def _fourier_batch_convolve_with_rir_padding(sound: torch.Tensor, rir: torch.Tensor):
    '''Convolve the sound with the rir using the fourier transform'''
    # rir has shape   (num_rirs, num_mics, num_sources, max_rir_length)
    # sound has shape (num_sounds, num_sources, max_sound_length)
    # we want to convolve each sound with all the rirs
    # output should have shape (num_sounds, num_rirs, num_mics, max_rir_length + max_sound_length - 1)

    num_sounds, num_sources, _ = sound.shape
    num_rirs, num_mics, _, max_rir_length = rir.shape
    output_time_shape = sound.shape[-1]

    rir = rir.view(num_rirs * num_mics, num_sources, max_rir_length)
    rir_fourier = torch.fft.rfft(rir, dim=-1, n=output_time_shape)
    sound_fourier = torch.fft.rfft(sound, dim=-1, n=output_time_shape)
    output_fourier = sound_fourier.unsqueeze(1)* rir_fourier.unsqueeze(0)
    output = torch.fft.irfft(output_fourier, n=output_time_shape, dim=-1)
    return output.view(num_sounds, num_rirs, num_mics, -1)[:,:,:,:output_time_shape]

class Rir:
    '''
    Class to compute the RIR (Room Impulse Response) and to convolve it with a given sound.
    '''
    def __init__(self, absorption_coefficient:Union[list,float], distance_from_wall: Union[list,float], room_side:float = 120.0, 
                 data_path:str = None, num_microphones:int=8,
                 use_cudnn:bool=True, no_fourier:bool=True) -> None:
        assert isinstance(absorption_coefficient, (list, float)) and isinstance(distance_from_wall, (list, float)), "absorption_coefficient and distance_from_wall must be either a list or a float"
        # absorption_coefficient and distance_from_wall must be either both lists or both floats
        if isinstance(absorption_coefficient, list):
            assert isinstance(absorption_coefficient, list) and isinstance(distance_from_wall, list) and len(absorption_coefficient) == len(distance_from_wall), "absorption_coefficient and distance_from_wall must be lists of the same length"
        else:
            assert isinstance(absorption_coefficient, float) and isinstance(distance_from_wall, float), "absorption_coefficient and distance_from_wall must be floats"
        
        self.absorption_coefficient = absorption_coefficient
        self.distance_from_wall = distance_from_wall
        self.room_side = room_side
        self.data_path = data_path
        
        if use_cudnn:
           torch.backends.cudnn.benchmark = True
           torch.backends.cuda.matmul.allow_tf32 = True
        else:
           torch.backends.cudnn.benchmark = False
        self.num_microphones = num_microphones
        self.no_fourier = no_fourier

    
    def convolve(self, sound:torch.Tensor):
        '''
        Convolve the given sound with the RIR.
        '''
        rir = self._get_rir()
        # rir can either have the shape (num_mics, num_sources, max_rir_length) or
        # (num_rirs, num_mics, num_sources, max_rir_length)
        # sound can either have the shape (num_sources, max_sound_length) or
        # (num_sounds, num_sources, max_sound_length)

        if len(rir.shape) == 4 or len(sound.shape) == 3:
            # unsqueeze the sound and rir to make sure they have the right shape
            if len(sound.shape) == 2:
                sound = sound.unsqueeze(0)
            if len(rir.shape) == 3:
                rir = rir.unsqueeze(0)
            rir = rir.to(sound.device)
            if self.no_fourier:
                return _batch_convolve_with_rir(sound=sound, rir=rir)
            else:
                return _fourier_batch_convolve_with_rir(sound=sound, rir=rir)
        output = Rir._convolve_with_rir(sound=sound, rir=rir)
        return torch.squeeze(output)

    def _get_rir(self):
        if isinstance(self.absorption_coefficient, list):
            rir = []
            for absorption_coefficient, distance_from_wall in zip(self.absorption_coefficient, self.distance_from_wall):
                if isinstance(absorption_coefficient, torch.Tensor):
                    absorption_coefficient = absorption_coefficient.item()
                if isinstance(distance_from_wall, torch.Tensor):
                    distance_from_wall = distance_from_wall.item()
                rir.append(self._get_rir_single(absorption_coefficient, distance_from_wall))
            # pad the rirs to the same length
            max_length = max([r.shape[-1] for r in rir])
            for i in range(len(rir)):
                rir[i] = torch.nn.functional.pad(rir[i], (0, max_length - rir[i].shape[-1]))
            rir = torch.stack(rir, dim=0)
        else:
            rir = self._get_rir_single(self.absorption_coefficient, self.distance_from_wall)
        return rir
    
    def _get_rir_single(self, absorption_coefficient:float, distance_from_wall:float):
        filename = f"rir_{Rir._num2str(absorption_coefficient)}_{Rir._num2str(distance_from_wall)}_{self.num_microphones}.npy"
        if self.data_path is not None:
            filename = os.path.join(data_path, filename)
        else:
            data_path = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "rir_bank")
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            filename = os.path.join(data_path, filename)
        
        if os.path.exists(filename):
            rir = np.load(filename)
            rir = torch.from_numpy(rir)
        else:
            rir = self._generate_rir(absorption_coefficient, distance_from_wall)
            np.save(filename, rir)
            rir = torch.from_numpy(rir)

        return rir

    @staticmethod
    def _num2str(num:float)->str:
        num_str = str(num)
        num_without_dot = num_str.replace(".", "dot")
        
        return num_without_dot
    
    @staticmethod
    def _str2num(s:str)->float:
        s = s.replace("dot", ".")
        return float(s)
    
    def _generate_rir(self, absorption_coefficient:float, distance_from_wall:float):
        '''
        Generate the RIR for the given absorption coefficient and distance from the wall.
        '''
        # code adapted from forward_model/inverse_problem_data_generation.py/generate_datapoint

        # Prepare arguments for the forward model
        args = []
        args.extend("-exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051".split())
        args.extend("-opt_harmonies 0.5 1 2 3".split())
        args.extend("-opt_harmonies_init_vals 0 0 0 0".split())
        args.extend("-opt_phi_0_init_vals 0 0 0 0".split())
        args.extend("-num_sources_power 7".split())
        args.extend("-channel 0".split())
        args.extend("-duration 0.5".split())
        args.extend("-num_rotors 4".split())
        args.extend(f"-e_absorption {absorption_coefficient}".split())
        args.extend("-mics_rads 0.53".split())
        args.extend("-max_order 1".split())

        args.extend(f"-num_phases_mics {self.num_microphones}".split())
        args.extend(f"-room_x {self.room_side}".split())
        args.extend(f"-room_y {self.room_side}".split())
        args.extend(f"-org_x {self.room_side/2}".split())
        args.extend(f"-org_y {distance_from_wall}".split())

        # add a flag to indicate that we only want to generate the RIR
        args.append("-rir_only")

        # build the parser and parse the arguments
        parser = argparse.ArgumentParser()
        arguments = add_args(parser, args)
        
        # run the forward model
        recordings = main_forward(arguments)
        print(f"Generated RIR with absorption coefficient {absorption_coefficient} and distance from wall {distance_from_wall} with shape {recordings.shape}")
        recordings = np.array(recordings)
        return recordings

    @staticmethod
    def _slow_batch_convolve_with_rir(sound:torch.Tensor, rir:torch.Tensor, num_sources=256,num_mics=512):
        '''DEPRECATED Convolve the sound with the rir'''
        # rir has shape   (num_rirs, num_mics, num_sources, max_rir_length)
        # sound has shape (num_sounds, num_sources, max_sound_length)
        # we want to convolve each sound with all the rirs
        # output should have shape (num_sounds, num_rirs, num_sources, num_mics, max_rir_length + max_sound_length - 1)
        raise NotImplementedError("This method is deprecated. Use _batch_convolve_with_rir instead.")
        # unsqueeze the sound and rir to make sure they have the right shape
        if len(sound.shape) == 2:
            sound = sound.unsqueeze(0)
        if len(rir.shape) == 3:
            rir = rir.unsqueeze(0)
        num_sounds = sound.shape[0]
        num_rirs = rir.shape[0]

        premix_signals = torch.zeros((num_sounds, num_rirs, num_sources, num_mics, sound.shape[-1]))
        for sound_idx in range(num_sounds):
            for rir_idx in range(num_rirs):
                for s in range(num_sources):
                    sig = sound[sound_idx, s]
                    h = rir[rir_idx, :, s]
                    conv_res = Rir._batch_convolve1d(sig, h)[0]
                    premix_signals[sound_idx, rir_idx, s, :, len(sig) + h.shape[1] - 1] += conv_res

        signals = torch.sum(premix_signals, axis=2)
        return signals
    
    @staticmethod
    def _convolve_with_rir(sound:torch.Tensor, rir:torch.Tensor, num_sources=256,num_mics=512):
        '''Convolve the sound with the rir'''
        premix_signals = torch.zeros((num_sources, num_mics, sound.shape[-1]))
        for s in range(num_sources):
            sig = sound[s]
            h = rir[:, s]
            conv_res = Rir._batch_convolve1d(sig, h)[0]
            # premix_signals = premix_signals.at[s, :, len(sig) + h.shape[1] - 1].add(
            #     conv_res)
            # the next line is a differentiable version of the following line
            premix_signals[s, :, len(sig) + h.shape[1] - 1] += conv_res


        signals = torch.sum(premix_signals, axis=0)
        return signals

    @staticmethod
    def _batch_convolve1d(x, y):
        # x: 1d tensor
        # y: batch of N 1d tensor
        # out: Nx(np.convolve(x,y[i]).shape[0])

        x_reshaped = x.reshape(1, 1, len(x))
        y_flipped = torch.flip(y.reshape(y.shape[0], 1, y.shape[1]), [2])
        padding = [y_flipped.shape[2] - 1]
        return torch.nn.functional.conv1d(x_reshaped, y_flipped, stride=1, padding=padding, dilation=1)


class Rir2d:
    '''
    Class to compute the RIR (Room Impulse Response) in the 2d settings and to convolve it with a given sound.
    '''
    def __init__(self, coordinates: Union[list, float],
                 data_path:str = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "rir", "rir_indoor_4_channels_5.0_5.0_order_1_0.5_d_0.05",
                                                        "rir_indoor"), use_cudnn:bool=True, no_fourier:bool = True) -> None:

        # ATTENTION: set data-path default to a different one

        # check data types
        assert isinstance(coordinates, (list, float))
        assert isinstance(data_path, str) or data_path is None
        assert isinstance(use_cudnn, bool)
        assert isinstance(no_fourier, bool)

        self.coordinates = coordinates
        self.data_path = data_path

        if use_cudnn:
           torch.backends.cudnn.benchmark = True
           torch.backends.cuda.matmul.allow_tf32 = True
        else:
           torch.backends.cudnn.benchmark = False
        
        self.no_fourier = no_fourier

    def convolve(self, sound:torch.Tensor):
        '''
        Convolve the sound with the RIR.
        :arg sound: the sound to convolve with the RIR. Must be a tensor of shape (num_sounds, num_sources, max_sound_length). or shape (num_sources, max_sound_length)
        '''
        # check data types
        assert isinstance(sound, torch.Tensor)
        assert len(sound.shape) in [2,3], "Sound must be a 2d or 3d tensor."

        if len(sound.shape) == 2:
            sound = sound.unsqueeze(0)

        rir = self._get_rir().to(sound.device)

        # convolve the sound with the RIR
        if self.no_fourier:
            return _batch_convolve_with_rir(sound=sound, rir=rir)
        else:
            return _fourier_batch_convolve_with_rir(sound=sound, rir=rir)
    
    def _get_rir(self):
        if isinstance(self.coordinates, float):
            return self._get_rir_single(self.coordinates)
        else:
            rir = [self._get_rir_single(c) for c in self.coordinates]
            # pad the rirs to the same length
            max_length = max([r.shape[-1] for r in rir])
            for i in range(len(rir)):
                rir[i] = torch.nn.functional.pad(rir[i], (0, max_length - rir[i].shape[-1]))
            return torch.stack(rir, dim=0)

    def _get_rir_single(self, coordinates:tuple):
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.tolist()
            for i in range(len(coordinates)):
                coordinates[i] = int(coordinates[i])
                
        filename = f"{coordinates[0]}_{coordinates[1]}.npy"
        path = os.path.join(self.data_path, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Could not find RIR file {path}")
        rir = torch.from_numpy(np.load(path))
        return rir
