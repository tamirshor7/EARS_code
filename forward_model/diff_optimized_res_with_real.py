#%%
import numpy as np
import matplotlib.pyplot as plt
from RealRecordings import RealRecordingsSeveralExperiments
import os, sys
import scipy
from scipy.optimize import curve_fit
cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from data_processing.create_signals_manually import plot_signals
from data_processing.pre_processing import *
# from data_processing.jax_curve_fit import jax_curve_fit
os.environ['CUDA_VISIBLE_DEVICES']= '2'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.20'
#%%
sim = np.load('sim.npy')
real = np.load('real.npy')
mx1=np.max(sim,1)
mx2=np.max(real,1)
res = mx1-mx2
print(f'max diff real vs sim: {max(np.abs(mx1-mx2))}')
# %%
radiuses = [str(x) for x in sorted(list(range(70,125,10)) + [75, 85])]
data_dir_names = ['13_9_'+str(x) for x in radiuses]
data_dir_paths = [os.path.join(cur_dir,'..','..','data','real_recordings', x) for x in data_dir_names]
recordings_mode = 'CCW'
vel_rate = 5
recordings_dataset = RealRecordingsSeveralExperiments(data_dir_paths, 
                                                        low_cut_coeff= -1,
                                                        high_cut_coeff= 3.25,
                                                        apply_filter=True,
                                                        filter_order=1,
                                                        rotor_type=18,
                                                        recordings_mode=recordings_mode, 
                                                        vel_rate=vel_rate)
# %%
duration = 0.6
real_recordings, encoder_readings, fs = recordings_dataset.get_avg_audio(duration)
omega = recordings_dataset.bpf
rps = recordings_dataset.rps

# %%
harmony = 0.5
order=3
filtered = butter_bandpass_filter(real_recordings, (harmony-0.1) * omega, (harmony+0.1) * omega, fs, order=order)
#filtered = butter_lowpass_filter(real_recordings, (1+0.25) * omega, fs, order=3)

idx_audio_channel_to_plot = 0
sig = real_recordings[idx_audio_channel_to_plot]
plt.plot(sig)
plt.grid()
plt.show()
#%%
# calc fft and show the magnitude spectrum of the signal
fft_sig = scipy.fftpack.rfft(sig)
xf = scipy.fftpack.rfftfreq(len(sig), 1 / fs)

bpf_harmonics = np.arange(rps*2, rps*10, rps*2)

plt.plot(xf, np.abs(fft_sig), zorder=1)
plt.scatter(bpf_harmonics, (np.abs(fft_sig))[8*bpf_harmonics.astype(np.int)], 
            s=80, facecolors='none', edgecolors='r', 
            label='BPF harmonics', zorder=2) #(np.abs(fft_sig))[bpf_harmonics.astype(np.int)], 
plt.scatter(rps, np.abs(fft_sig)[8*int(rps)], 
            s=80, facecolors='none', edgecolors='green', 
            label='RPS', zorder=2)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [-]')
plt.legend()
plt.xlim(left=0, right=500)
plt.title(f'Magnitude Spectrum of audio channel {idx_audio_channel_to_plot-1}, RPS={round(rps,2)}[Hz]')
plt.show()
plt.clf()


# mul_coeff = 100
# plt.scatter(recordings_dataset.mics_dist * mul_coeff, np.max(filtered[:,500:2000],1), label='real', color='orange')
# plt.show() 
# plt.clf()
# # %%
# def func(x, a, b):
#     return a / x**b

# mul_coeff = 100
# radiuses_float = recordings_dataset.mics_dist[::4] * mul_coeff
# popt, pcov = curve_fit(func, radiuses_float, np.max(sim, 1)[::4],
#                         bounds=([0,2], [np.inf,4]))
# x_range = np.linspace(min(radiuses_float), max(radiuses_float), 500)
# # x_range = np.linspace(min(all_mics_poses_sorted)-5, max(all_mics_poses_sorted)+5, 500)
# plt.plot(x_range, func(x_range, *popt), label=f'Fitted to `{round(popt[0],2)} / x**{round(popt[1],2)}`', color='purple', zorder=1, lw=0.5)
# plt.scatter(recordings_dataset.mics_dist * mul_coeff, np.max(real,1), label='real', color='orange')
# plt.scatter(radiuses_float, np.max(sim, 1)[::4], label='sim', color='purple')
# plt.xlabel('mics dist [cm]')
# plt.legend()
# plt.show()
# plt.clf()
#%%
def func(x, a, b):
    return a / x**b

mul_coeff = 100
radiuses_float_all = recordings_dataset.mics_dist * mul_coeff
radiuses_float = radiuses_float_all[::4]
popt, pcov = curve_fit(func, radiuses_float_all, jnp.max(real_recordings, 1),
                        bounds=([0,2], [jnp.inf,4]))
x_range = jnp.linspace(min(radiuses_float), max(radiuses_float), 500)
plt.plot(x_range, func(x_range, *popt), label=f'Real fitted to `{round(popt[0],2)} / x**{round(popt[1],2)}`', color='orange', zorder=1, lw=0.5)

plt.scatter(recordings_dataset.mics_dist * mul_coeff, jnp.max(real_recordings,1), label='real', color='orange')
plt.xlabel('mics dist [cm]')
plt.legend()
plt.show()
plt.clf()
# %%
