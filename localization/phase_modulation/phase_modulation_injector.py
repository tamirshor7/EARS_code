import torch
#import numpy as np
from typing import Union

def _match_time_dimension(phase_modulation:torch.Tensor, original_sound_time_dimension:int) -> torch.Tensor:
    # make the time dimension of original_sound and phase modulation match
    if phase_modulation.shape[-1] < original_sound_time_dimension:
        extension = int(torch.ceil(torch.tensor(original_sound_time_dimension/phase_modulation.shape[-1])))
        phase_modulation = torch.tile(phase_modulation, (1, extension))
        phase_modulation = phase_modulation[...,:original_sound_time_dimension]
    elif phase_modulation.shape[-1] > original_sound_time_dimension:
        phase_modulation = phase_modulation[..., :original_sound_time_dimension]
    return phase_modulation

def _build_indexer_over_rotors(batch_size:int, num_rotors:int, number_time_samples:int, device:torch.device) -> torch.Tensor:
    # Make sure not to interpolate between the rotors by using indexer_rotors
    indexer_rotors = torch.linspace(-1,1, num_rotors, dtype=torch.float64, device=device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    indexer_rotors = torch.tile(indexer_rotors, (batch_size, 1, number_time_samples, 1))
    return indexer_rotors

def _restrict_to_its_period(phase_modulation:torch.Tensor, period:Union[float,int]) -> torch.Tensor:
    phase_modulation = torch.remainder(phase_modulation, period) #phase_modulation % period
    # TO CANCEL
    if not (phase_modulation.min()>=0 and phase_modulation.max() <= period):
        breakpoint()
    #assert phase_modulation.min()>=0 and phase_modulation.max() <= period, f"Attention the restriction produced out of bounds values! (original phase modulation: {original_phase_modulation} period: {period} phase modulation {phase_modulation})"
    return phase_modulation

def _build_identity_modulation(num_rotors:int, number_time_samples:int, dt:float, device:torch.device) -> torch.Tensor:
    identity_modulation = torch.linspace(0, dt*number_time_samples, number_time_samples, dtype=torch.float64, device=device).unsqueeze(0)
    identity_modulation = torch.tile(identity_modulation, (num_rotors, 1))
    return identity_modulation

def _build_omega(omega_per_rotor:float, device:torch.device) -> torch.Tensor:
    # Take into account the fact that some rotors are spinning clockwise and some counterclockwise (hence the minus sign)
    # Plus sign is for the counter clockwise rotors, minus sign for the clockwise rotors
    omega = torch.tensor(
        [[-omega_per_rotor], [omega_per_rotor], 
         [omega_per_rotor], [-omega_per_rotor]], dtype=torch.float64, device=device)
    # shape (4,1)
    return omega

def _project_phase_modulation(phase_modulation:torch.Tensor, batch_size:int, dt:float) -> torch.Tensor:
    # assert that phase modulation is in its period which is [0, dt*phase_modulation.shape[-1]]
    assert phase_modulation.min() >= 0 and phase_modulation.max() <= dt*phase_modulation.shape[-1], "Attention the projection received a phase modulation out of bounds!"

    # Transform the phase modulation from its period (which is [0, dt*phase_modulation.shape[-1]]) to 
    # being normalized in [-1,1] as expected by grid_sample
    phase_modulation = torch.tile(phase_modulation.unsqueeze(0).unsqueeze(-1), (batch_size, 1, 1,1))/(dt*phase_modulation.shape[-1])*2-1
    assert phase_modulation.min()>=-1 and phase_modulation.max() <= 1
    
    return phase_modulation

def _apply_modulation(original_sound:torch.Tensor, phase_modulation:torch.Tensor, omega_per_rotor:float=46.9333688083713, dt:float=1/3003.735603735763, interpolation_mode:str='bilinear', use_radians:bool=True) -> torch.Tensor:
    '''
    Inject phase modulation in sound
    :param original_sound: sound in which to inject the phase modulation
                           expected shape: (batch_size, microphone, rotor, time)
    :param phase_modulation: tensor used to modulate original_sound
                           expected shape: (rotor, time)
    :param omega_per_rotor: the angular velocity [rad/s] with which each rotor is spinning 
    :param use_radians: if True phase_modulation is expressed in radians, else in degrees
    '''
    # control the shape and differentiability of original_sound and of phase_modulation
    assert len(original_sound.shape) == 4, "The original sound must be 4D: (batch_size, microphone, rotor, time)"
    #assert original_sound.requires_grad, "Please set original_sound.requires_grad = True"
    assert original_sound.dtype == torch.float64, "In order to reduce numerical errors cast original sound to the torch.float64 dtype"

    assert len(phase_modulation.shape) == 2, "The phase modulation must be 2D: (rotor, time)"
    #assert phase_modulation.requires_grad, "Please set phase_modulation.requires_grad = True"
    assert phase_modulation.dtype == torch.float64, "In order to reduce numerical errors cast phase_modulation to the torch.float64 dtype"
    
    assert original_sound.shape[2] == phase_modulation.shape[0], f"original sound and phase modulation have a different rotor dimension: original sound {original_sound.shape[2]} phase modulation {phase_modulation.shape[0]}"
    assert original_sound.device == phase_modulation.device , f"Original sound is in device {original_sound.device} and phase modulation is in device {phase_modulation.device}. Please move them to the same device"
    
    is_training = phase_modulation.requires_grad
    
    # Reshape phase_modulation so that it complies with grid_sample
    batch_size, num_rotors, number_time_samples = original_sound.shape[0], original_sound.shape[2], original_sound.shape[3]
    assert num_rotors == 4, f"Currently we only support 4 rotors, you passed {num_rotors}"
    total_time_in_seconds = dt*number_time_samples

    phase_modulation = _match_time_dimension(phase_modulation=phase_modulation, original_sound_time_dimension=number_time_samples)

    indexer_rotors = _build_indexer_over_rotors(batch_size, num_rotors, number_time_samples, device=phase_modulation.device)

    if not use_radians:
        phase_modulation = torch.deg2rad(phase_modulation)

    #phase_modulation = _restrict_to_its_period(phase_modulation, period=2*torch.pi)
    
    # Given that this is the phase modulation, it represents the difference between an identity modulation
    # and the actual modulation we want to get, hence we need to add the identity modulation!
    identity_modulation = _build_identity_modulation(num_rotors=num_rotors, number_time_samples=number_time_samples, dt=dt, device=phase_modulation.device)
    omega = _build_omega(omega_per_rotor=omega_per_rotor, device=phase_modulation.device)
    phase_modulation = identity_modulation - phase_modulation/omega
    phase_modulation = _restrict_to_its_period(phase_modulation, period=total_time_in_seconds)

    # Transform the phase modulation from being in radians to being normalized in [-1,1] with circular boundary conditions
    phase_modulation = _project_phase_modulation(phase_modulation=phase_modulation, batch_size=batch_size, dt=dt)
    #print(f'first samples of phase modulation {phase_modulation[0,0,835:845].squeeze()} (shape {phase_modulation.shape})')
    grid = torch.zeros(batch_size, num_rotors, number_time_samples, 2, dtype=torch.float64, device=phase_modulation.device)
    grid[:,:,:,1] = indexer_rotors.squeeze()
    grid[:,:,:,0] = phase_modulation.squeeze()

    # Use nearest to not interpolate between the rotors
    # Use bicubic to have smoother gradients but slightly interpolate between the rotors
    modulated_sound = torch.nn.functional.grid_sample(original_sound, grid, align_corners=True, mode=interpolation_mode)

    #assert original_sound.requires_grad, "Attention there is an operation that breaks differentiability in original sound"
    #assert phase_modulation.requires_grad, "Attention there is an operation that breaks differentiability in phase modulation"
    
    assert not (is_training and not modulated_sound.requires_grad), "Attention there is an operation that breaks differentiability in modulated sound"
    
    return modulated_sound

@torch.jit.script
def apply_modulation(original_sound:torch.Tensor, phase_modulation:torch.Tensor, omega_per_rotor:float=46.9333688083713, dt:float=1/3003.735603735763, use_radians:bool=True, interpolation_mode:str='bilinear'):
    '''
    Inject phase modulation in sound
    :param original_sound: sound in which to inject the phase modulation
                           expected shape: (batch_size, rotor, microphone, time)
    :param phase_modulation: tensor used to modulate original_sound
                           expected shape: (rotor, time)
    :param omega_per_rotor: the angular velocity [rad/s] with which the rotors are spinning 
    :param use_radians: if True phase_modulation is expressed in radians, else in degrees
    '''
    assert interpolation_mode in ['bicubic','bilinear', 'nearest'], f"You chose a wrong interpolation mode ({interpolation_mode}). Please choose among bicubic, bilinear and nearest"
    # currently the input has shape (batch_size, rotor, microphone, time) and we convert it to (batch_size, microphone, rotor, time)
    original_sound = torch.permute(original_sound, (0,2,1,3))
    #original_sound.requires_grad = True
    output_sound = _apply_modulation(original_sound, phase_modulation, omega_per_rotor=omega_per_rotor, dt=dt, use_radians=use_radians, interpolation_mode=interpolation_mode)
    # output has shape (batch_size, microphone, rotor, time) and we need to sum over the rotor dimension
    output_sound = torch.sum(output_sound, 2)
    return output_sound
