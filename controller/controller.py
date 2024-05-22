import control as ctrl
from EARS.localization.physics import PLOT_DT
import numpy as np
import torch

def snr_db_to_variance(snr_db, signal_power):
    # signal_power is the norm of the signal
    return signal_power / (np.power(10, snr_db / 10))

class System(torch.nn.Module):
    def __init__(self, recording_length, K_vco=1,Kp=200,Ki=0,Kd=50, dt=PLOT_DT) -> None:
        super().__init__()
        self.dt = dt
        self.time = np.arange(recording_length)*self.dt
        self.K_vco = K_vco
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.plant = ctrl.TransferFunction([K_vco], [1, 0])
        self.pid_controller = ctrl.TransferFunction([Kp+Kd, Kp+Ki, Ki], [1, 1, 0])
        self.system_pll_to_reference = ctrl.feedback(self.pid_controller*self.plant)
        self.system_pll_to_noise = ctrl.feedback(self.plant, self.pid_controller)
    def forward(self, reference_signal, system_noise_snr_db:float=float('inf')):
        orig_signal = torch.clone(reference_signal)
        reference_signal = reference_signal.detach().cpu().numpy()
        _, response_to_reference = ctrl.forced_response(self.system_pll_to_reference, self.time, reference_signal)
        noise_gain = snr_db_to_variance(system_noise_snr_db, np.linalg.norm(response_to_reference))
        white_noise = ctrl.white_noise(self.time, Q=noise_gain, dt=self.dt)
        _, response_to_noise = ctrl.forced_response(self.system_pll_to_noise, self.time, white_noise[0])
        response = response_to_reference + response_to_noise
        with torch.no_grad():
            orig_signal.copy_(torch.tensor(response))
            
        return orig_signal
    def backward(self,grad_output):
        return grad_output

class FastSystem(torch.nn.Module):
    def __init__(self, recording_length, K_vco=1,Kp=200,Ki=0,Kd=50, dt=PLOT_DT,
                 device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) -> None:
        super().__init__()
        self.dt = dt
        self.time = np.arange(recording_length)*self.dt
        self.K_vco = K_vco
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.plant = ctrl.TransferFunction([K_vco], [1, 0])
        self.pid_controller = ctrl.TransferFunction([Kp+Kd, Kp+Ki, Ki], [1, 1, 0])
        self.system_pll_to_reference = ctrl.feedback(self.pid_controller*self.plant)
        _, self.system_pll_to_reference_impulse_response = ctrl.impulse_response(self.system_pll_to_reference, self.time)
        self.system_pll_to_reference_impulse_response = torch.as_tensor(self.system_pll_to_reference_impulse_response, device=device)
        self.system_pll_to_reference_impulse_response = self.system_pll_to_reference_impulse_response.flip(0).unsqueeze(0).unsqueeze(0)

        self.system_pll_to_noise = ctrl.feedback(self.plant, self.pid_controller)
        _, self.system_pll_to_noise_impulse_response = ctrl.impulse_response(self.system_pll_to_noise, self.time)
        self.system_pll_to_noise_impulse_response = torch.as_tensor(self.system_pll_to_noise_impulse_response, device=device)
        self.system_pll_to_noise_impulse_response = self.system_pll_to_noise_impulse_response.flip(0).unsqueeze(0).unsqueeze(0)
        
    def forward(self, reference_signal, system_noise_snr_db:float=float('inf')):
        reference_signal = torch.nn.functional.pad(reference_signal, (self.system_pll_to_reference_impulse_response.shape[-1]-1,0)).unsqueeze(1)
        response_to_reference = torch.nn.functional.conv1d(reference_signal, self.system_pll_to_reference_impulse_response, padding=0).squeeze()*self.dt
        signal_power = torch.linalg.norm(response_to_reference).detach().cpu().numpy()

        noise_gain = snr_db_to_variance(system_noise_snr_db, signal_power)
        Q = np.diag(noise_gain*np.ones(reference_signal.shape[0]))
        white_noise = ctrl.white_noise(self.time, Q=Q, dt=self.dt)
        white_noise = torch.as_tensor(white_noise, device=self.system_pll_to_noise_impulse_response.device).unsqueeze(1)
        white_noise = torch.nn.functional.pad(white_noise, (self.system_pll_to_noise_impulse_response.shape[-1]-1,0))
        
        response_to_noise = torch.nn.functional.conv1d(white_noise, self.system_pll_to_noise_impulse_response, padding=0).squeeze()*self.dt

        response = response_to_reference + response_to_noise
        return response

class FastFourierSystem(torch.nn.Module):
    def __init__(self, recording_length, K_vco=1,Kp=200,Ki=0,Kd=50, dt=PLOT_DT,
                 device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) -> None:
        super().__init__()
        self.dt = dt
        self.time = np.arange(recording_length)*self.dt
        self.K_vco = K_vco
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.plant = ctrl.TransferFunction([K_vco], [1, 0])
        self.pid_controller = ctrl.TransferFunction([Kp+Kd, Kp+Ki, Ki], [1, 1, 0])
        self.system_pll_to_reference = ctrl.feedback(self.pid_controller*self.plant)
        _, self.system_pll_to_reference_impulse_response = ctrl.impulse_response(self.system_pll_to_reference, self.time)
        self.system_pll_to_reference_impulse_response = torch.as_tensor(self.system_pll_to_reference_impulse_response, device=device)
        self.system_pll_to_reference_impulse_response = self.system_pll_to_reference_impulse_response.unsqueeze(0)
        self.system_pll_to_reference_impulse_response = torch.fft.rfft(self.system_pll_to_reference_impulse_response, dim=-1, n=self.time.shape[-1])

        self.system_pll_to_noise = ctrl.feedback(self.plant, self.pid_controller)
        _, self.system_pll_to_noise_impulse_response = ctrl.impulse_response(self.system_pll_to_noise, self.time)
        self.system_pll_to_noise_impulse_response = torch.as_tensor(self.system_pll_to_noise_impulse_response, device=device)
        self.system_pll_to_noise_impulse_response = self.system_pll_to_noise_impulse_response.unsqueeze(0)
        self.system_pll_to_noise_impulse_response = torch.fft.rfft(self.system_pll_to_noise_impulse_response, dim=-1, n=self.time.shape[-1])

    def forward(self, reference_signal, system_noise_snr_db:float=float('inf')):
        reference_signal = torch.fft.rfft(reference_signal.squeeze(), dim=-1, n=self.time.shape[-1])
        response_to_reference = torch.fft.irfft(reference_signal*self.system_pll_to_reference_impulse_response, dim=-1, n=self.time.shape[-1]).squeeze()*self.dt
        signal_power = torch.linalg.norm(response_to_reference).detach().cpu().numpy()

        noise_gain = snr_db_to_variance(system_noise_snr_db, signal_power)
        Q = np.diag(noise_gain*np.ones(reference_signal.shape[0]))
        white_noise = ctrl.white_noise(self.time, Q=Q, dt=self.dt)
        white_noise = torch.as_tensor(white_noise, device=self.system_pll_to_noise_impulse_response.device).unsqueeze(1)
        white_noise = torch.fft.rfft(white_noise.squeeze(), dim=-1, n=self.time.shape[-1])
        
        response_to_noise = torch.fft.irfft(white_noise*self.system_pll_to_noise_impulse_response, dim=-1, n=self.time.shape[-1]).squeeze()*self.dt
        

        response = response_to_reference + response_to_noise
        return response
