import torch
from torch import nn
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir, '..', '..', '..')
sys.path.insert(0, parent_dir)
from EARS.localization.transformer_encoder import SpectrogramTransformer, SpectrogramCNN, \
    WideSpectrogramCNN, AggregateTransformer, SoundTransformer, OrientationAggregateTransformer, \
    OrientationSpectrogramTransformer, OrientationAggregateTransformerMLP, OrientationAggregateTransformerAttention
import numpy as np
import matplotlib.pylab as P
from EARS.localization.preprocess import preprocess_audio_waveform
import math
from EARS.localization.phase_modulation.phase_modulation_injector import apply_modulation
from EARS.localization.physics import PLOT_DT
from EARS.controller.controller import System, FastSystem, FastFourierSystem

def snr_db_to_variance(snr_db, signal_power):
    # signal_power is the norm of the signal
    return signal_power / (torch.pow(torch.tensor(10), snr_db / 10))



def plot_phases(phases, name):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(nrows=2, ncols=2)

    for i in range(phases.shape[0]):
        row = i // 2  # Row index of the subplot
        col = i % 2  # Column index of the subplot
        ax = axes[row, col]  # Get the corresponding subplot
        ax.plot(range(phases.shape[-1]), phases[i, :].cpu().detach())  # Plot the data on the subplot
        ax.set_title(f"Rotor {i}")
        ax.set_ylabel("Phase Shift")
        ax.set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(f"{name}.png")


class PhaseMod(nn.Module):

    def basis2curve(self):
        if not self.use_fourier:
            raise("basis to curve conversion is only supported in fourier basis mode")

        return torch.sum(self.amplitudes*(self.basis.to(self.amplitudes.device)), dim=1)

    def basis2curve_factored(self):
        if not self.use_fourier:
            raise("basis to curve conversion is only supported in fourier basis mode")

        raise ValueError("basis2curve_factored is deprecated")

    def initialize_phases(self, recording_length, initialization='radial', n_rotors=4, learn_shifts=True, use_fourier=True,\
                          cutoff = None, basis_size = None,device = "cuda",
                          use_cosine:bool=False, init_size = 1):


        self.recording_length = 2 ** math.floor(math.log2(recording_length)) + 1
        
        dt = PLOT_DT

        if use_fourier:
            '''Get all harmonics. We model basis_size harmonics each with freq of tau*cutoff/T, s.t tau are the 
            harmonics and T is the period imposed by the modeling window of 8 revolutions 1025*dt = 0.34 secs'''

            period_time = self.recording_length*dt

            phi_dot_freqs_raw = (torch.arange(self.recording_length)* dt *\
                                 torch.arange(1,basis_size+1).reshape(basis_size, 1)) #frequencies for derivative (velocity)

            #cutoff*tau*t/T
            phi_dot_freqs = phi_dot_freqs_raw* (2*torch.pi*cutoff/period_time)


            '''basis family of cosines for phi_dot (derivative of phase = velocity) - integrated for sine basis for 
            phases. cos(tau*cutoff*x/T)-> T/(cutoff*tau)sin(tau*cutoff*x/T). Constant coefficients are ignored since
            they are absorbed by learned amplitudes'''

            if use_cosine:
                self.basis = (torch.cos(phi_dot_freqs)/torch.arange(1,basis_size+1).unsqueeze(-1)).unsqueeze(0)\
                    .repeat(n_rotors,1,1).to(device).to(torch.float64) #curve per rotor
            else:
                self.basis = (torch.sin(phi_dot_freqs)/torch.arange(1,basis_size+1).unsqueeze(-1)).unsqueeze(0)\
                    .repeat(n_rotors,1,1).to(device).to(torch.float64) #curve per rotor

            self.amplitudes = torch.zeros(n_rotors, basis_size, 1, device=device)

            if initialization == "sine":
                self.amplitudes[:,0,0] = init_size

            if initialization =="grad_freq_sine":
                self.amplitudes[0, 0, 0] = init_size
                self.amplitudes[1, 1, 0] = init_size
                self.amplitudes[2, 2, 0] = init_size
                self.amplitudes[3, 3, 0] = init_size

            elif initialization =="max_freq_const":
                self.amplitudes[:, -1, 0] = init_size
            
            elif initialization == "antiphase_sines":
                for i in range(n_rotors):
                    self.amplitudes[i, :, 0] = init_size*dt*torch.sin(self.basis_size*torch.arange(0,self.basis_size,1) + i*torch.pi/2)
            
            elif initialization == "antiphase_time_domain":
                self.amplitudes[0, 0, 0] = init_size
                self.amplitudes[1, 0, 0] = init_size
                self.amplitudes[2, 0, 0] = init_size
                self.amplitudes[3, 0, 0] = init_size
                for i in range(n_rotors):
                    self.basis[i] = (torch.sin(2*phi_dot_freqs+ (torch.ones(10,)*i*torch.pi/2).view(-1,1).repeat(1,phi_dot_freqs.shape[-1])))

            self.amplitudes = nn.Parameter(self.amplitudes, requires_grad=learn_shifts)

            self.x = self.basis2curve()




        elif initialization == 'radial':
            x = torch.zeros(n_rotors, self.recording_length)
            theta = np.pi / n_rotors
            for i in range(n_rotors):
                Ly = torch.arange(-np.pi, np.pi, 2 * np.pi / self.recording_length).float()
                x[i, :] = Ly * np.sin(theta * i)

        elif initialization == 'sine':
            x = torch.zeros(n_rotors, self.recording_length)
            theta = np.pi / n_rotors
            for i in range(n_rotors):
                Ly = torch.sin(torch.arange(-np.pi, np.pi, 2 * np.pi / self.recording_length).float())
                x[i, :] = Ly  
        elif initialization == 'multisine':
            freq = 2
            x = torch.zeros(n_rotors, self.recording_length)
            theta = np.pi / n_rotors
            for i in range(n_rotors):
                Ly = torch.sin(freq * torch.arange(-np.pi, np.pi, 2 * np.pi / self.recording_length).float())
                x[i, :] = Ly  

        elif initialization == 'sincos':
            x = torch.zeros(n_rotors, self.recording_length)
            theta = np.pi / n_rotors
            for i in range(n_rotors):
                Ly = torch.sin(
                    (i % 2 * (np.pi / 2)) + torch.arange(-np.pi, np.pi, 2 * np.pi / self.recording_length).float())
                x[i, :] = Ly  


        elif initialization == 'uniform':
            x = (torch.rand(n_rotors, self.recording_length) - 0.5) * self.res
        elif initialization == 'gaussian':
            x = torch.randn(n_rotors, self.recording_length) * self.res / 6
        elif initialization == 'zero':
            x = torch.zeros(n_rotors, self.recording_length)
        elif initialization == 'constant':
            x = torch.ones(n_rotors, self.recording_length) * np.pi / 2
        else:
            raise ValueError('Wrong initialization')

        if not use_fourier:
            self.x = torch.nn.Parameter(x, requires_grad=learn_shifts)

    def __init__(self, decimation_rate, rec_len, res, learn_shifts, initialization, n_rotors, interp_gap, max_vel,
                 max_acc, \
                 projection_iterations=10e2, project=False, device='cuda' if torch.cuda.is_available() else 'cpu',\
                 use_fourier=True,cutoff = None, basis_size = None,
                 use_cosine:bool=False,init_size = 1):
        super().__init__()

        self.decimation_rate = decimation_rate
        self.res = res
        self.num_measurements = res ** 2 // decimation_rate
        self.use_fourier = use_fourier
        self.basis_size = basis_size
        self.initialize_phases(rec_len, initialization, n_rotors, learn_shifts,self.use_fourier,cutoff,self.basis_size,
                               use_cosine=use_cosine, init_size = init_size)
        self.project = project
        self.interp_gap = interp_gap
        self.device = device
        self.iters = projection_iterations
        self.max_vel = max_vel
        self.max_acc = max_acc

    def forward(self):
        #The forward shapes the phase curve to match kinematic constraints
        if self.use_fourier:
            self.x = self.basis2curve()

        # interpolate
        if self.interp_gap > 1:
            t = torch.arange(0, self.x.shape[1], device=self.x.device).float()
            t1 = t[::self.interp_gap]
            x_short = self.x[:, ::self.interp_gap]     

            for rotor in range(x_short.shape[0]):
                self.x.data[rotor, :] = self.interp(t1, x_short[rotor, :], t)
                

        return self.x

    def get_phases(self):
        return self.basis2curve() 

    def h_poly(self, t):
        tt = [None for _ in range(4)]
        tt[0] = 1
        for i in range(1, 4):
            tt[i] = tt[i - 1] * t
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], dtype=tt[-1].dtype)
        return [
            sum(A[i, j] * tt[j] for j in range(4))
            for i in range(4)]

    def interp(self, x, y, xs):
        m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
        I = P.searchsorted(x[1:].detach().cpu(), xs.detach().cpu())
        dx = (x[I + 1] - x[I])
        hh = self.h_poly((xs - x[I]) / dx)
        return hh[0] * y[I] + hh[1] * m[I] * dx + hh[2] * y[I + 1] + hh[3] * m[I + 1] * dx

    def __repr__(self):
        return f'Phase_Shift_Layer'


class Localization_Model(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, num_microphones, dropout, decimation_rate, res,
                 learn_shift, initialization, n_rotors, interp_gap, rec_len, max_vel, max_acc,
                 projection_iters=10e2, project=False, device='cuda' if torch.cuda.is_available() else 'cpu',
                 no_fourier: bool = False, inject_noise: bool = False, standardize: bool = False,
                 num_coordinates: int = 1,
                 use_cnn: bool = False, use_wide_cnn: bool = False, 
                 use_fourier=True, cutoff = None, basis_size = None,
                 use_cosine:bool = False, high_resolution_stft:bool = False,
                 sampling_interpolation_method:str = 'bilinear',
                 phase_modulation_snr_db_list:list = [25,30,35],init_size =1,
                 use_aggregate:bool = False, use_sound_transformer:bool = False,
                 use_orientation_aggregate:bool = False, use_orientation_transformer:bool = False,
                 simulate_system:bool = False, use_orientation_aggregate_mlp:bool = False,
                 use_orientation_aggregate_attention:bool = False,
                 simulate_system_method:str = "original"):
        assert not (use_cnn and use_wide_cnn), "You can either choose a CNN or a wide CNN but not both"
        super().__init__()
        self.device = device
        self.no_fourier = no_fourier
        self.inject_noise = inject_noise
        self.standardize = standardize

        self.high_resolution_stft = high_resolution_stft

        self.sampling_interpolation_method = sampling_interpolation_method

        self.phase_modulation_snr_db_list = phase_modulation_snr_db_list

        assert num_coordinates in [1,2,3]
        self.num_coordinates = num_coordinates

        self.phase_model = PhaseMod(decimation_rate, rec_len, res, learn_shift, initialization, n_rotors,
                                    interp_gap, max_vel, max_acc, projection_iters, project, device=device,\
                                    use_fourier=use_fourier,cutoff=cutoff, basis_size=basis_size,
                                    use_cosine=use_cosine, init_size = init_size)
        if use_orientation_aggregate:
            if use_orientation_transformer:
                self.backward_model = OrientationSpectrogramTransformer(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout,
                                                                  num_coordinates=self.num_coordinates)
            elif use_orientation_aggregate_mlp:
                self.backward_model = OrientationAggregateTransformerMLP(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout,
                                                                  num_coordinates=self.num_coordinates)
            elif use_orientation_aggregate_attention:
                self.backward_model = OrientationAggregateTransformerAttention(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout,
                                                                               num_coordinates=self.num_coordinates)
            else:
                self.backward_model = OrientationAggregateTransformer(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout,
                                                                    num_coordinates=self.num_coordinates)
        elif use_cnn:
            self.backward_model = SpectrogramCNN(output_coordinates=self.num_coordinates)
        elif use_wide_cnn:
            self.backward_model = WideSpectrogramCNN(hidden_dim=hidden_dim, num_coordinates=self.num_coordinates)
        elif use_aggregate:
            self.backward_model = AggregateTransformer(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout,
                                                        num_coordinates=self.num_coordinates)
        elif use_sound_transformer:
            self.backward_model = SoundTransformer(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout,
                                                        num_coordinates=self.num_coordinates)
        else:
            self.backward_model = SpectrogramTransformer(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout,
                                                        num_coordinates=self.num_coordinates)
        self.orig_state_dict = self.backward_model.state_dict()

        if simulate_system:
            supported_methods = ["original", "time", "fourier"]
            assert simulate_system_method in supported_methods, f"Please set simulate_system_method to one of {supported_methods}. Got {simulate_system_method}"
            if simulate_system_method == "original":
                self.system = System(self.phase_model.recording_length)
            elif simulate_system_method == "time":
                self.system = FastSystem(self.phase_model.recording_length)
            elif simulate_system_method == "fourier":
                self.system = FastFourierSystem(self.phase_model.recording_length)
            
        else:
            self.system = None
    def get_system_response(self, phases: torch.Tensor, system_noise_snr_db: float=float('inf'), omega_per_rotor:float=46.9333688083713,
                            dt:float = PLOT_DT):
        assert self.system is not None, f"Can't compute the response of the system when self.system is None. Please set simulate_system to True in the constructor"
        if isinstance(self.system, FastSystem) or isinstance(self.system, FastFourierSystem):
            return self.new_get_system_response(phases, system_noise_snr_db, omega_per_rotor, dt=PLOT_DT)
        injected_phase_modulation = []
        for i in range(4):
            if i in [0,3]:
                ref = phases[i] - omega_per_rotor*torch.arange(self.phase_model.recording_length, device=phases.device)*dt
            else:
                ref = phases[i] + omega_per_rotor*torch.arange(self.phase_model.recording_length, device=phases.device)*dt
                
            injected =  self.system(phases[i].cpu(), system_noise_snr_db=system_noise_snr_db).to(phases.device)

            injected_phase_modulation.append(injected)
            phases[i] = self.system(ref.cpu(), system_noise_snr_db=system_noise_snr_db).to(phases.device)
        injected_phase_modulation = torch.stack(injected_phase_modulation)
        return phases, injected_phase_modulation
    
    def new_get_system_response(self, phases: torch.Tensor, system_noise_snr_db: float=float('inf'), omega_per_rotor:float=46.9333688083713,
                            dt:float = PLOT_DT):
        assert self.system is not None, f"Can't compute the response of the system when self.system is None. Please set simulate_system to True in the constructor"
        constant_speed_effect = (omega_per_rotor*torch.arange(self.phase_model.recording_length, device=phases.device)*dt).unsqueeze(0).repeat(4,1)
        mask = torch.tensor([-1, 1, 1, -1], device=phases.device)
        constant_speed_effect = constant_speed_effect*mask.unsqueeze(-1)
        ref = phases + constant_speed_effect
        
        injected_phase_modulation = self.system(phases, system_noise_snr_db=system_noise_snr_db)
        phases = self.system(ref, system_noise_snr_db=system_noise_snr_db)
        return phases, injected_phase_modulation

    def forward(self, input, injected_phases=None, desired_snr_in_db:float=None, orientation=None, system_noise_snr_db:float=float('inf')):
        phases = self.phase_model()
        if injected_phases is None:
            injected_phases = phases
        elif self.system is not None:
            raise NotImplementedError("We didn't handle the case in which we both inject gaussian noise in the input and we simulate the PID response")

        if self.inject_noise:

            chosen_snr_db = np.random.choice(self.phase_modulation_snr_db_list)
            chosen_variance = snr_db_to_variance(chosen_snr_db, torch.linalg.norm(injected_phases, dim=-1))
            chosen_variance = torch.unsqueeze(chosen_variance, dim=-1)
            injected_phases = injected_phases + torch.randn_like(injected_phases) * chosen_variance
        


        omega_per_rotor = 2*np.pi*23.46668440418565

        dt = PLOT_DT

        if self.system is not None:
            if isinstance(self.system, FastSystem) or isinstance(self.system, FastFourierSystem):
                phases, injected_phases = self.new_get_system_response(phases, system_noise_snr_db, omega_per_rotor, dt=PLOT_DT)
            else:
                phases, injected_phases = self.get_system_response(phases, system_noise_snr_db, omega_per_rotor, dt=PLOT_DT)
        
        input = input.to(torch.float64)

        output_sound = apply_modulation(input, injected_phases, omega_per_rotor=omega_per_rotor,
                                         dt=dt, interpolation_mode=self.sampling_interpolation_method)
        

        if desired_snr_in_db is not None:
            variance = snr_db_to_variance(desired_snr_in_db, torch.linalg.norm(output_sound, dim=-1))
            variance = torch.unsqueeze(variance, dim=-1)
            output_sound = output_sound + torch.randn_like(output_sound)*variance

        if self.high_resolution_stft:
            n_fft = 400
            hop_length = 100
        else:
            n_fft = 40
            hop_length = 20
        output_sound = preprocess_audio_waveform(output_sound, n_fft = n_fft, hop_length = hop_length, standardize=self.standardize)

        
        phases = preprocess_audio_waveform(phases, n_fft = n_fft, hop_length = hop_length, standardize=self.standardize)
        if orientation is not None:
            orientation = torch.stack((torch.cos(orientation), torch.sin(orientation)),dim=-1)
            estimated_distance = self.backward_model(output_sound, phases, orientation)
        else:
            assert not isinstance(self.backward_model, OrientationAggregateTransformer), "OrientationAggregateTransformer expects to receive orientation, but orientation was set to None!"
            estimated_distance = self.backward_model(output_sound, phases)
        return estimated_distance

    def get_phases(self):
        return self.phase_model.get_phases()
    
    def recons_reset(self):
        self.backward_model.load_state_dict(self.orig_state_dict)