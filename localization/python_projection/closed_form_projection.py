import numpy as np
from scipy.linalg import solve_banded
import torch

class ClosedFormProjector:
    """
    Closed form projection of a phase to a velocity and acceleration constraint
    """

    def __init__(self, maximum_velocity:float, maximum_acceleration:float, signal_length:int,
                 device='cuda' if torch.cuda.is_available() else 'cpu') -> None:
        assert maximum_velocity is not None and maximum_acceleration is not None, "maximum_velocity and maximum_acceleration must be specified"
        assert isinstance(maximum_velocity, float) and isinstance(maximum_acceleration, float), "maximum_velocity and maximum_acceleration must be floats"
        self.maximum_velocity = maximum_velocity
        self.maximum_acceleration = maximum_acceleration
        
        assert signal_length is not None, "signal_length must be specified"
        assert isinstance(signal_length, int), "signal_length must be an integer"
        assert signal_length > 0, "signal_length must be positive"
        self.signal_length = signal_length

        self.device = device
        
        self._compute_constraints()
    
    def __repr__(self) -> str:
        return f"ClosedFormProjector(velocity_constraint={self.velocity_constraint}, acceleration_constraint={self.acceleration_constraint})"
    
    def _compute_constraints(self) -> None:
        # Velocity constraint
        velocity_constraint = get_velocity_matrix_constraint(self.signal_length, self.maximum_velocity)
        self.velocity_constraint = torch.from_numpy(velocity_constraint).to(self.device)

        # Acceleration constraint
        acceleration_constraint = solve_banded((1,1), get_banded_acceleration_matrix(self.signal_length), get_constraint(self.signal_length, self.maximum_acceleration))
        self.acceleration_constraint = torch.from_numpy(acceleration_constraint).to(self.device)

    def __call__(self, phase:torch.Tensor) -> torch.Tensor:
        """
        Closed form projection of a phase to a velocity and acceleration constraint
        """
        assert isinstance(phase, torch.Tensor), "Phase must be a torch tensor"
        assert phase.shape[-1] == self.signal_length, f"Phase must have shape (..., {self.signal_length})"
        assert phase.device == self.device, f"Phase must be on device {self.device}"

        phase = torch.clamp(phase, -self.velocity_constraint, self.velocity_constraint)
        phase = torch.clamp(phase, -self.acceleration_constraint, self.acceleration_constraint)

        return phase

# utility functions

def get_banded_acceleration_matrix(n):
    return np.stack([
        np.concatenate([np.ones(n-1), np.zeros(1)]), 
        -2*np.ones(n), 
        np.concatenate([np.ones(n-1), np.zeros(1)])
        ])

def get_banded_velocity_matrix(n):
    raise NotImplementedError("This function is not implemented yet")
    return np.stack([
        np.concatenate([np.ones(n-1), np.zeros(1)]), 
        -np.ones(n), 
        np.concatenate([np.zeros(1), np.ones(n-1)])
        ])

def get_velocity_matrix_constraint(n, val):
    return (n-np.arange(1,n+1)+1)*val*np.ones(n)

def get_constraint(n, val):
    return val*np.ones(n)

def project(phase, velocity_constraint, acceleration_constraint):
    """
    Deprecated: use ClosedFormProjector
    Closed form projection of a phase to a velocity and acceleration constraint
    """
    n = phase.shape[-1]
    # Velocity projection
    
    # Classical projection
    # velocity_constraint = solve_banded((1,1), get_banded_velocity_matrix(n), get_constraint(n, velocity_constraint))
    # velocity_constraint = torch.from_numpy(velocity_constraint).to(phase.device)
    # projected_phase = torch.clamp(phase, -velocity_constraint, velocity_constraint)

    # Closed form projection
    velocity_constraint = get_velocity_matrix_constraint(n, velocity_constraint)
    velocity_constraint = torch.from_numpy(velocity_constraint).to(phase.device)
    projected_phase = torch.clamp(phase, -velocity_constraint, velocity_constraint)

    # acceleration projection
    acceleration_constraint = solve_banded((1,1), get_banded_acceleration_matrix(n), get_constraint(n, acceleration_constraint))
    acceleration_constraint = torch.from_numpy(acceleration_constraint).to(phase.device)
    projected_phase = torch.clamp(projected_phase, -acceleration_constraint, acceleration_constraint)

    return projected_phase