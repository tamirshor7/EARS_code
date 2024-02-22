import torch
import numpy as np
#from EARS.localization.physics import number_of_revolutions_to_number_of_raw_samples

# ATTENTION: If you call these functions directly, make sure that the parameters are correct, they might be different from the default!
RECORDING_LENGTH: int = 1_501
BASIS_SIZE: int = 100
CUTOFF: float = 10.0

class Penalty:
    def __init__(self, physics) -> None:
        # physics is an instance of the Physics class
        self.physics = physics
    def _fwhm_to_std(self,fwhm:list, device='cpu'):
        fwhm = [self.physics.number_of_revolutions_to_number_of_raw_samples(f) for f in fwhm]
        return torch.tensor(fwhm, device=device)*(2*np.sqrt(2 * np.log(2)))

    def _get_batched_filter(self,fwhm_list, device, basis_size=BASIS_SIZE, cutoff=CUTOFF, rec_len=RECORDING_LENGTH,
                            num_rotors=4):
        # final shape: (num_fwhm, 1, 1)
        std_values = self._fwhm_to_std(fwhm_list, device).unsqueeze(-1).unsqueeze(-1)
        
        # Relevant frequencies
        # final shape: (1, num_rotors, basis_size)
        #x_values = (torch.arange(basis_size, device=device)*2*torch.pi*cutoff/(rec_len-1)).repeat(1,num_rotors, 1)
        #x_values = (torch.arange(basis_size, device=device) * 2 * torch.pi * cutoff).repeat(1, num_rotors,1)#unit conversion factored version
        #x_values = torch.arange(1,basis_size+1, device=device).repeat(1, num_rotors,1)#unit conversion factored version
        
        x_values = torch.arange(1,basis_size+1, device=device).repeat(1, num_rotors,1)


        # final shape: (num_fwhm, num_rotors, basis_size)
        kernel_values = torch.exp(-x_values**2/(2*std_values**2))
        normalized_kernel_values = kernel_values/torch.sum(kernel_values, dim=-1, keepdim=True)
        # final shape: (num_fwhm, num_rotors, basis_size, 1)
        normalized_kernel_values = normalized_kernel_values.reshape(len(fwhm_list), num_rotors, basis_size, 1)
        
        return normalized_kernel_values

    def get_integrated_velocity_penalty(self,amplitudes, fwhm_list, basis_size=BASIS_SIZE, cutoff=CUTOFF, rec_len=RECORDING_LENGTH,
                                        num_rotors=4):
        batched_kernel_list = self._get_batched_filter(fwhm_list, device=amplitudes.device, basis_size=basis_size, cutoff=cutoff, rec_len=rec_len,
                                                num_rotors=num_rotors)
        # coefficients shape: (1, num_rotors, basis_size, 1)
        coefficients = torch.unsqueeze(amplitudes**2, dim=0)
        # pairwise difference of penalty over counterclockwise and clockwise rotors
        rotor_directions = torch.tensor([-1,1,1,-1], device=coefficients.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        penalty = torch.abs(torch.sum(batched_kernel_list*coefficients*rotor_directions))

        return penalty