from abc import ABC, abstractmethod
import torch
import numpy as np
import os

class PhaseModulator(ABC):
    def __init__(self, sound_params:dict = None) -> None:
        super().__init__()
        '''
        Sound parameters is a dictionary with the following keys:
        omega: angular velocity of the sound
        coeffs: optimized coefficients of the harmonies
        phi_0: initial phase of the harmonies
        harmonies: harmonies of the sound
        fs: sampling frequency
        duration: duration of the sound
        num_rotors: number of rotors in the sound
        '''
        self.sound_params = sound_params

    @staticmethod
    def _generate_input_sound(modulation, omega, coeffs, phi_0=0, harmonies=[1], fs=44100, duration=0.5, num_rotors=1, signals_len=None):
        # Code readapted from EARS/data_processing/create_signals_manually.py function create_harmonic_signal_matrix_style_with_coeff
        # TODO: Check if this new implementation works! (Hence modulation will have a shape of (batch_size, num_sources, num_phases_mics)

        # ATTENTION: We are assuming that in case of batching, the shape of modulation will be:
        # (batch_size, num_rotors, time_steps)
        if not len(modulation.shape) in [2, 3]:
            raise ValueError(f'Unrecognized shape of modulation. Expected 2 or 3, got {len(modulation.shape)}')
        
        # in phase model use_batches is False
        use_batches = len(modulation.shape) == 3

        if signals_len is None:
            signals_len = int(duration * fs)
        
        # t = jnp.expand_dims(jnp.linspace(0, duration, signals_len), 0)
        t = torch.unsqueeze(torch.linspace(0, duration, signals_len), 0)
        
        # harmonies = jnp.array(harmonies)
            
        if len(coeffs) != len(harmonies):
            raise ValueError(f'Unmatching dims of coeffs. coeffs {len(coeffs)} vs harmonies {len(harmonies)}')
        
        # harmonies = harmonies[:, jnp.newaxis]
        # harmonies_omega_t = jnp.tile(harmonies * 2 * jnp.pi * omega * t, (phi_0.shape[1],1,1))
        harmonies = torch.unsqueeze(harmonies, 1)
        phi_0 = phi_0.to(modulation.device)
        coeffs = coeffs.to(modulation.device)
        if not use_batches:
            # shape is (num_sources_per_rotor*num_rotors, num_harmonies, signals_len)
            harmonies_omega_t = torch.tile(harmonies * 2 * np.pi * omega * t, (phi_0.shape[1],1,1)).to(modulation.device)
            # in our case is 256
            num_sources = harmonies_omega_t.shape[0]//num_rotors
            # shape is (num_sources_per_rotor, num_harmonies, signals_len)
            phase_modulation_single_rotor = torch.tile(modulation, (num_sources,1,1))
        else:
            harmonies_omega_t = torch.tile(harmonies * 2 * np.pi * omega * t, (1, phi_0.shape[1],1,1)).to(modulation.device)
            num_sources = harmonies_omega_t.shape[1]//num_rotors
            phase_modulation_single_rotor = torch.tile(modulation, (1, num_sources,1,1))
        

        # phase_modulation_single_rotor = jnp.tile(phase_modulation, (num_sources,1,1))
        # phase_modulation = jnp.zeros_like(harmonies_omega_t)

        # phase_modulation has shape (num_sources_per_rotor*num_rotors, num_harmonies, signals_len)
        phase_modulation = torch.zeros_like(harmonies_omega_t, requires_grad=True).to(modulation.device)
        
        # More general version of the above code (no need to check if we are using batches or not)
        for i in range(num_rotors):
            # The following is a differentiable version of this instruction
            # phase_modulation[:,i*num_sources:(i+1)*num_sources,:,:] = phase_modulation_single_rotor
            phase_modulation = torch.cat((phase_modulation[...,:i*num_sources,:,:], phase_modulation_single_rotor, phase_modulation[...,(i+1)*num_sources:,:,:]))

        # TODO: Why transpose?
        # TODO: Check the correctness of the following code!
        # cos_res = jnp.cos(harmonies_omega_t + jnp.expand_dims(phi_0.T,2) + phase_modulation)
        if use_batches:
            phi_0 = torch.unsqueeze(torch.unsqueeze(phi_0.T, 2), 0)
            cos_res = torch.cos(harmonies_omega_t + phi_0 + phase_modulation)
        else:
            cos_res = torch.cos(harmonies_omega_t + torch.unsqueeze(phi_0.T,2) + phase_modulation)

        # samples shape is num_radiuses_circles*num_sources_in_circle,1
        # samples = jnp.squeeze((jnp.expand_dims(coeffs.T,1) @ cos_res), 1)
        # TODO: Check the correctness of the following code!
        if use_batches:
            coeffs = torch.unsqueeze(torch.unsqueeze(coeffs.T, 1), 0)
            samples = torch.squeeze((coeffs @ cos_res), 1)
        else:
            samples = torch.squeeze((torch.unsqueeze(coeffs.T,1) @ cos_res), 1)
        return samples

    @staticmethod
    def _get_input_sound(modulation:torch.Tensor, sound_params:dict = None):
        '''Generate the input sound that needs to be convolved with the rir'''

        # Load default parameters
        if sound_params is None:
            sound_params = dict(np.load(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "input_sound_bank", "input_sound_parameters.npz")))
            sound_params['signals_len'] = None

        omega = torch.tensor(sound_params['omega'])
        coeffs = torch.tensor(sound_params['coeffs'])
        phi_0 = torch.tensor(sound_params['phi_0'])
        harmonies = torch.tensor(sound_params['harmonies'])
        fs = sound_params['fs']
        duration = float(sound_params['duration'])
        num_rotors = sound_params['num_rotors']
        signals_len = sound_params['signals_len']
        
        # Make the size of the modulation match the size of the sound
        if signals_len is None:
            audio_len = int(duration * fs)
        else:
            audio_len = signals_len

        if modulation.shape[-1] <= audio_len:
            # if the modulation is equal or smaller, repeat it
            modulation = torch.tile(modulation, (1, audio_len//modulation.shape[-1] + 1))
            modulation = modulation[:,:audio_len]
        elif modulation.shape[-1] > audio_len:
            # if the modulation is bigger, cut it (generate a warning)
            UserWarning(f"The modulation's temporal length ({modulation.shape[-1]}) is longer than the sound's temporal length ({audio_len}), the modulation will be cut to match the length of the sound!")
            modulation = modulation[...,:audio_len]

        input_modulated_sound = PhaseModulator._generate_input_sound(modulation=modulation, omega=omega, coeffs=coeffs, phi_0=phi_0, harmonies=harmonies, fs=fs, duration=duration, num_rotors=num_rotors,
                                                                     signals_len=signals_len)
        return input_modulated_sound
    
    def get_input_sound(self):
        return PhaseModulator._get_input_sound(modulation=self.modulation, sound_params=self.sound_params)

    def get_input_sound_and_modulation(self, declared_modulation:torch.Tensor = None):
        '''To declare a modulation different from the one that is actually used specify it in the declared_modulation parameter.'''
        modulation = self.get_modulation()
        assert modulation.requires_grad, 'The modulation must be differentiable'
        sound = PhaseModulator._get_input_sound(modulation=modulation, sound_params=self.sound_params)
        if declared_modulation is not None:
            modulation = declared_modulation
        else:
            modulation = self.get_modulation()

        assert len(sound) == len(modulation), f'len(sound) {len(sound)} != len(modulation) {len(modulation)}'

        return sound, modulation

    @abstractmethod
    def get_modulation(self):
        ...

class PhaseModulatorRaw(PhaseModulator):
    '''The modulation is represented as a torch Tensor of shape (4,?) where:
    - the first axis corresponds to a rotor
    - the second axis corresponds to the ? samples of the modulation
    
    The modulation is not generated from a basis, it is directly given by the user.
    It can be useful to test the performance of the algorithm with a known modulation or to use a modulation that is not representable by a basis, 
    e.g. a modulation outputted by a neural network.
    '''
    def __init__(self, modulation, sound_params:dict = None,
                 duration=None, signals_len=None) -> None:
        if (duration is not None) and (signals_len is not None) and (sound_params is None):
            sound_params = dict(np.load(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "input_sound_bank", "input_sound_parameters.npz")))
            sound_params['duration'] = duration
            sound_params['signals_len'] = signals_len
        super().__init__(sound_params)
        self.modulation = modulation
    
    def get_modulation(self):
        return self.modulation

class PhaseModulatorFromBasis(PhaseModulator):
    def __init__(self, coefficients, sound_params:dict = None) -> None:
        super().__init__(sound_params)
        self.coefficients = coefficients
        self.coefficients.requires_grad = True

    def generate_modulation_from_basis(self):
        return PhaseModulator._generate_modulation_from_basis(self.coefficients)

    @staticmethod
    @abstractmethod
    def generate_modulation_from_basis(coefficients):
        ...

    def get_modulation(self):
        modulation = self.generate_modulation_from_basis()
        assert modulation.requires_grad, 'The modulation must be a tensor with requires_grad=True'
        return modulation

    def get_input_sound_and_modulation(self, declared_modulation = None):
        '''To declare a modulation different from the one that is actually used specify it in the declared_modulation parameter.'''
        modulation = self.get_modulation()
        assert modulation.requires_grad, 'The modulation must be differentiable'
        sound = PhaseModulator._get_input_sound(modulation=modulation, sound_params=self.sound_params)
        if declared_modulation is not None:
            if isinstance(declared_modulation, torch.Tensor):
                # TODO: Are we sure that the declared modulation is differentiable?
                declared_modulation.requires_grad = True
                modulation = declared_modulation
            else:
                modulation = PhaseModulatorFromBasis.generate_modulation_from_basis(declared_modulation)
        else:
            modulation = self.get_modulation()

        assert len(sound) == len(modulation), f'len(sound) {len(sound)} != len(modulation) {len(modulation)}'

        return sound, modulation
    
class PhaseModulatorConstant(PhaseModulatorFromBasis):
    '''The coefficients are represented as a torch Tensor of shape (4,1)'''
    def __init__(self, coefficients) -> None:
        PhaseModulatorConstant._check_validity(coefficients)
        super().__init__(coefficients)
    
    @staticmethod
    def generate_modulation_from_basis(coefficients):
        PhaseModulatorConstant._check_validity(coefficients)
        return coefficients

    @staticmethod
    def _check_validity(coefficients):
        assert isinstance(coefficients, torch.Tensor), f'coefficients must be a torch Tensor, not {type(coefficients)}'
        assert 2 <= len(coefficients.shape) <= 3 and coefficients.shape[-2:] == (4,1), f'coefficients must be a torch Tensor of shape (batch,4,1) or (4,1), not {coefficients.shape}'
        # check that not every coefficient is equal
        assert not torch.equal(coefficients, coefficients[0]), f'All the coefficients are equal to {coefficients[0]}, please change at least one coefficient'

class PhaseModulatorLinear(PhaseModulatorFromBasis):
    '''The coefficients are represented as a torch Tensor of shape (4,2) where:
    - the first axis corresponds to a rotor
    - the second axis corresponds to the slope and the intercept of the linear function
    '''
    def __init__(self, coefficients) -> None:
        assert isinstance(coefficients, torch.Tensor), f'coefficients must be a torch Tensor, not {type(coefficients)}'
        assert coefficients.shape == (4,2), f'coefficients must be a torch Tensor of shape (4,2), not {coefficients.shape}'
        super().__init__(coefficients)
    
    @staticmethod
    def generate_modulation_from_basis(coefficients):
        assert isinstance(coefficients, torch.Tensor), f'coefficients must be a torch Tensor, not {type(coefficients)}'
        assert coefficients.shape == (4,2), f'coefficients must be a torch Tensor of shape (4,2), not {coefficients.shape}'
        
        # TODO: implement
        raise NotImplementedError('The generate_modulation_from_basis function is not implemented yet')

class PhaseModulatorSpline(PhaseModulatorFromBasis):
    '''The coefficients are represented as a torch Tensor of shape (4,?) where:
    - the first axis corresponds to a rotor
    - the second axis corresponds to the ? coefficients of the spline function
    '''
    # TODO: implement
    pass