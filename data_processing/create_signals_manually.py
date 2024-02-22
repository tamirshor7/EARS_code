#%%
from jax import numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import os
import scipy
import pdb

#%%
def calculate_rps(enc_channel, sampling_rate, bpf = False):
    fft_enc = scipy.fftpack.rfft(enc_channel)
    xf = scipy.fftpack.rfftfreq(len(enc_channel), 1 / sampling_rate)
    if bpf:
        #return xf[np.argmax(np.abs(fft_enc))] / 2
        plt.plot(xf, jnp.abs(fft_enc))
        plt.xlim(left=0, right=200)
        plt.show()
    return xf[jnp.argmax(jnp.abs(fft_enc))] * 2 # we multiply the max value by two, because of our signal's characteristic


#%%
def create_harmonic_signal(omega, phi_0=0, max_harmony=2, fs=44100, duration=0.5):
    samples = jnp.zeros(int(fs * duration))
    t = jnp.linspace(0, duration, int(duration * fs))
    for k in range(1, max_harmony+1):
        samples += (1/k) * jnp.cos( k * 2 * jnp.pi * omega  * t + phi_0)
    return samples

#%%
def create_harmonic_signal_matrix_style(omega, phi_0=0, max_harmony=2, fs=44100, duration=0.5):
    t = jnp.expand_dims(jnp.linspace(0, duration, int(duration * fs)), 0)
    harmonies = jnp.array(range(1, max_harmony+1))
    harmonies_omega_t = jnp.expand_dims(harmonies, 1) * omega * t 
    cos_res = jnp.cos(harmonies_omega_t * 2 * jnp.pi + phi_0)
    samples = (1 / harmonies) @ cos_res
    return samples
    
# %% 
# FIXME: it is fixed with phase shifts. Now need to fix all other functions!
# FIXME: Also, in all other functions change max_harmony to harmonies

def create_harmonic_signal_matrix_style_with_coeff_old(omega, coeffs, phi_0=0, harmonies=[1], fs=44100, duration=0.5):
    t = jnp.expand_dims(jnp.linspace(0, duration, int(duration * fs)), 0)
    
    harmonies = jnp.array(harmonies)
        
    if len(coeffs) != len(harmonies):
        raise ValueError(f'Unmatching dims of coeffs. coeffs {len(coeffs)} vs harmonies {len(harmonies)}')
    
    harmonies = harmonies[:, jnp.newaxis]

    cos_res = jnp.cos(harmonies * 2 * jnp.pi * omega * t + phi_0)
    samples = coeffs @ cos_res
    return samples

def create_harmonic_signal_matrix_style_with_coeff(omega, coeffs, phi_0=0, harmonies=[1], fs=44100, duration=0.5, modulate_phase=None, num_rotors=1,
                                                   phase_modulation_injected=None):
    signals_len = int(duration * fs)
    t = jnp.expand_dims(jnp.linspace(0, duration, signals_len), 0)
    
    harmonies = jnp.array(harmonies)
        
    if len(coeffs) != len(harmonies):
        raise ValueError(f'Unmatching dims of coeffs. coeffs {len(coeffs)} vs harmonies {len(harmonies)}')
    
    harmonies = harmonies[:, jnp.newaxis]
    harmonies_omega_t = jnp.tile(harmonies * 2 * jnp.pi * omega * t, (phi_0.shape[1],1,1))
    #breakpoint()
    if modulate_phase is None:
        # cos_res shape is num_radiuses_circles*num_sources_in_circle, num_harmonies,1
        if phase_modulation_injected is None:
            cos_res = jnp.cos(harmonies_omega_t + jnp.expand_dims(phi_0.T,2))
        else:
            if phase_modulation_injected.shape[-1] <= signals_len:
                # tile
                phase_modulation_injected = jnp.tile(phase_modulation_injected, (1,signals_len//phase_modulation_injected.shape[-1]+1))
                phase_modulation_injected = phase_modulation_injected[..., :signals_len]
            else:
                phase_modulation_injected = phase_modulation_injected[..., :signals_len]
            cos_res = jnp.cos(harmonies_omega_t + jnp.expand_dims(phi_0.T,2)+jnp.expand_dims(phase_modulation_injected,0))
    else:
        magic_number = 5 * 2 # revolutions #TODO: times 2 because we want to not count the first x revolutions
        spin_frac_of_sig = int(2*fs*magic_number/omega)
        num_sources = harmonies_omega_t.shape[0]//num_rotors
        #phase_modulation = jnp.tile(jnp.linspace(0, jnp.pi, spin_frac_of_sig), int(signals_len/spin_frac_of_sig)+1)[:signals_len]
        phase_modulation = jnp.tile(jnp.linspace(-jnp.pi, jnp.pi, spin_frac_of_sig), int(signals_len/spin_frac_of_sig)+1)[:signals_len] # TODO: from -pi to pi
        phase_modulation_single_rotor = jnp.tile(phase_modulation, (num_sources,1,1))
        no_modulation_single_rotor = jnp.zeros((num_sources, 1, harmonies_omega_t.shape[2]))
        phase_modulation = jnp.zeros_like(harmonies_omega_t)
        # ATTENTION: this if-else is not in the original code and it is added to support the case when modulate_phase is not a list
        if num_rotors == 1:
            if modulate_phase:
                phase_modulation = phase_modulation.at[0:num_sources,:,:].set(phase_modulation_single_rotor)
            else:
                phase_modulation = phase_modulation.at[0:num_sources,:,:].set(no_modulation_single_rotor)
        else:
            for i in range(num_rotors):
                print(modulate_phase)
                if modulate_phase[i]:
                    phase_modulation = phase_modulation.at[i*num_sources:(i+1)*num_sources,:,:].set(phase_modulation_single_rotor)
                else:
                    phase_modulation = phase_modulation.at[i*num_sources:(i+1)*num_sources,:,:].set(no_modulation_single_rotor)
        cos_res = jnp.cos(harmonies_omega_t + jnp.expand_dims(phi_0.T,2) + phase_modulation)
    # samples shape is num_radiuses_circles*num_sources_in_circle,1
    samples = jnp.squeeze((jnp.expand_dims(coeffs.T,1) @ cos_res), 1)
    return samples
#%%
def create_harmonic_signal_with_phase_modulation(omega, max_harmony=2, odd=True, fs=44100, duration=0.5):
    samples = jnp.zeros(int(fs * duration))
    for k in range(1, max_harmony+1):
        # if (odd and k%2 == 1) or (not odd and k%2 == 0):
        samples += (1/k) * jnp.cos(  k * 2 * jnp.pi * omega  * jnp.linspace(0, duration, int(duration * fs))  + jnp.pi/2 +
                                jnp.cos(omega * k * jnp.linspace(0, duration, int(duration * fs))))
    return samples

# %%
def create_harmonic_signal_with_phase_modulation_matrix_style(omega, phi_0=jnp.pi/2, max_harmony=2, fs=44100, duration=0.5):
    t = jnp.expand_dims(jnp.linspace(0, duration, int(duration * fs)), 0)
    harmonies = jnp.array(range(1, max_harmony+1))
    harmonies_omega_t = jnp.expand_dims(harmonies, 1) * omega * t
    cos_res = jnp.cos(2 * jnp.pi * harmonies_omega_t + jnp.cos(harmonies_omega_t) + phi_0)
    samples = (1 / harmonies) @ cos_res
    return samples

# %%
def cretae_single_harmonic_signal_with_phase(omega, A0, phi_0=jnp.pi/2, harmonic=1, fs=44100, duration=0.5):
    t = jnp.linspace(0, duration, int(duration * fs))
    harmonic_omega_t = harmonic * omega * t
    cos_res = A0 * jnp.cos(2 * jnp.pi * harmonic_omega_t + jnp.cos(harmonic_omega_t) + phi_0)
    return cos_res

#%%
def plot_signals(signals, fs, signals_labels=None, title=None, save_to_file=None):
    if signals_labels is not None and len(signals_labels) != signals.shape[0]:
        raise ValueError("Unmatching shapes of signals and their labels.")

    x_range = jnp.arange(signals.shape[1]) / fs #jnp.arange(fs*duration) / fs

    for i in range(signals.shape[0]):
        if signals_labels is not None:
            plt.plot(x_range, signals[i], label=signals_labels[i])
        else:
            plt.plot(x_range, signals[i])
    
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    if title is not None:
        plt.title(title)
    if save_to_file is None:
        plt.show()
    else:
        plt.savefig(save_to_file)
    plt.clf()
# %%
if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES']= '1'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.20'

    fs=44100
    duration=0.5
    x_range = jnp.arange(fs*duration) / fs

    #samples = create_harmonic_signal(omega=30, max_harmony=2, phi_0=jnp.pi/2)
    #plt.plot(x_range, samples)

    # samples = create_harmonic_signal_with_phase_modulation(omega=30, max_harmony=4)
    # plt.plot(samples)
    """
    samples = create_harmonic_signal_matrix_style(omega=30, max_harmony=1, phi_0=0)
    plt.plot(x_range, samples)

    samples = create_harmonic_signal_matrix_style(omega=30, max_harmony=1, phi_0=jnp.pi)
    plt.plot(x_range, samples)
    """
    #samples = create_harmonic_signal_matrix_style(omega=30, max_harmony=1, phi_0=0)
    #plt.plot(x_range, samples)



    # samples = create_harmonic_signal_with_phase_modulation_matrix_style(omega=30, max_harmony=4)
    # plt.plot(samples)

    # max_harmony = 7
    # coeffs = jnp.random.rand(max_harmony)
    # coeffs = jnp.array([1,0.5])
    # coeffs = jnp.array([[-0.05, -0.07, -0.07],
    #                     [-0.05, -0.05, -0.05]])
    coeffs = jnp.array([-0.05, -0.07])
    phi_0 = jnp.expand_dims(jnp.array([0, jnp.pi]),1)
    # phi_0 = jnp.array([[0, 0, 0],
    #                     [jnp.pi, jnp.pi, jnp.pi]])
    harmonies = [1, 0.5]
    fs = 3003
    samples = create_harmonic_signal_matrix_style_with_coeff_old(omega=23, coeffs=coeffs, harmonies=harmonies, phi_0=phi_0, fs=fs)
    coeffs = jnp.array([[-0.05, -0.05],
                        [-0.05, -0.05],
                        [-0.05, -0.05]]).T
    phi_0 = jnp.array([[0, 0, 0],
                       [0, 0, 0]])
    samples1 = create_harmonic_signal_matrix_style_with_coeff(omega=23, coeffs=coeffs, harmonies=harmonies, phi_0=phi_0, fs=fs)
    phi_0 = jnp.ones_like(phi_0) * jnp.pi/2
    samples2 = create_harmonic_signal_matrix_style_with_coeff(omega=23, coeffs=coeffs, harmonies=harmonies, phi_0=phi_0, fs=fs)#, modulate_phase=True)

    #plot_signals(jnp.expand_dims(samples1, axis=0), fs, ['Sig1'], 'Signals')
    #plot_signals(jnp.asarray([samples, samples2]), fs, ['Sig1', 'Sig2'], 'Signals')
    plt.plot(samples1[0])
    plt.plot(samples2[0])

# %%
    harmonies = [1]
    fs = 3003
    coeffs = jnp.array([[-0.05],
                        [-0.05],
                        [-0.05]]).T
    phi_0 = jnp.array([[0, 0, 0]])
    samples1 = create_harmonic_signal_matrix_style_with_coeff(omega=23, coeffs=coeffs, harmonies=harmonies, phi_0=phi_0, fs=fs)
    #phi_0 = jnp.array([[jnp.pi/2, 0, 0]])
    samples2 = create_harmonic_signal_matrix_style_with_coeff(omega=23, coeffs=coeffs, harmonies=harmonies, phi_0=phi_0, fs=fs, modulate_phase=True)

    plt.plot(samples2[0])
    plt.plot(samples1[0])

# %%
