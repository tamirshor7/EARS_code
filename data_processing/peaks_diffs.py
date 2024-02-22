#%% imports
import numpy as np
import os, sys

# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import pandas as pd

from scipy.optimize import curve_fit

from pre_processing import *
from utils import *

#%% define params
fs = 44100
mode = 'CCW' 
rotor_type = 16
num_velocities = 15
chosen_audio_channels = [0,1,2,3]
compare_by_energy = False # if true - compensate the signals by the diff in mics in energy (avg_sig**2), else by max amplitude of the avg sig
apply_fit = True # if true - fit curve to `a / (x**(b)`
specific_vels = None # a list of velocities to plot together with fitted curves. If None, than plotting all velocities
compensate_diffs = False

#%% read data from h5

data_exp_name = '15_7_exp_wav'
data_dir_path = os.path.join('..','..','data','real_recordings',data_exp_name)
data_file_path = os.path.join(data_dir_path, 'real_recordings.h5')
df_data = pd.read_hdf(data_file_path)
mics_data, mics_pos, mics_distances_sorted, mics_distances_sorted_idx = read_mics_metadata(data_dir_path, mode)

data_exp_name1 = '13_7_SD_exp_wav'
data_dir_path = os.path.join('..','..','data','real_recordings',data_exp_name1)
data_file_path = os.path.join(data_dir_path, 'real_recordings.h5')
df_data1 = pd.read_hdf(data_file_path)
mics_data1, mics_pos1, mics_distances_sorted1, mics_distances_sorted_idx1 = read_mics_metadata(data_dir_path, mode)

data_exp_name2 = '13_7_LD_exp_wav'
data_dir_path = os.path.join('..','..','data','real_recordings',data_exp_name2)
data_file_path = os.path.join(data_dir_path, 'real_recordings.h5')
df_data2 = pd.read_hdf(data_file_path)
mics_data2, mics_pos2, mics_distances_sorted2, mics_distances_sorted_idx2 = read_mics_metadata(data_dir_path, mode)

#%% plot peaks for each velocity
powers = []
covs = []
def func(x, a, b):
    return a / x**b

for vel_rate in list(range(0,6))+list(range(7,num_velocities)):

    avg_audio_all_np, maxes, mins, integrals, rps = comp_avg_sig( df_data, chosen_audio_channels, mode, rotor_type, vel_rate, fs)
    avg_audio_all_np1, maxes1, mins1, integrals1, rps1 = comp_avg_sig( df_data1, chosen_audio_channels, mode, rotor_type, vel_rate, fs)
    avg_audio_all_np2, maxes2, mins2, integrals2, rps2 = comp_avg_sig( df_data2, chosen_audio_channels, mode, rotor_type, vel_rate, fs)

    diffs_coeff_peak =  maxes[2] / maxes
    diffs_coeff_integrals =  integrals[2] / integrals
    
    if compensate_diffs:
        maxes1 = maxes1 * diffs_coeff_peak
        integrals1 = integrals1 * diffs_coeff_integrals

    maxes1 = maxes1[mics_distances_sorted_idx1]
    integrals1 = integrals1[mics_distances_sorted_idx1]
    mins1 = mins1[mics_distances_sorted_idx1]

    if compensate_diffs:
        maxes2 = maxes2 * diffs_coeff_peak
        integrals2 = integrals2 * diffs_coeff_integrals

    maxes2 = maxes2[mics_distances_sorted_idx2]
    integrals2 = integrals2[mics_distances_sorted_idx2]
    mins2 = mins2[mics_distances_sorted_idx2]

    if compare_by_energy:
        compenstaed_sig1 = integrals1
        compenstaed_sig2 = integrals2
        title_text = f'Audio peaks VS mics angular coords, rotor {rotor_type} {mode}\ncomparison by energy'
        #ylabel_text = r'Energy $\int{avgSig ^ 2}$'
        ylabel_text = 'Energy=integrate(avg_sig ^ 2)'
        #plt.ylabel(ylabel_text)
    else: # by max amplitude
        compenstaed_sig1 = maxes1
        compenstaed_sig2 = maxes2
        title_text = f'Audio peaks VS mics angular coords, rotor {rotor_type} {mode}\ncomparison by max amplitude'
        ylabel_text = 'Max audio amplitude'

    all_mics_pos = np.concatenate((mics_distances_sorted1, mics_distances_sorted2))
    all_mics_poses_sorted = np.sort(all_mics_pos)
    all_mics_poses_sorted_idx = np.argsort(all_mics_pos)

    all_compensted_sigs = np.concatenate((compenstaed_sig1, compenstaed_sig2))[all_mics_poses_sorted_idx]


    if specific_vels is not None:
        if apply_fit and vel_rate in specific_vels:
            popt, pcov = curve_fit(func, all_mics_poses_sorted, all_compensted_sigs, 
                                bounds=([0,1.5], [np.inf,4]))
                      
            x_range = np.linspace(min(all_mics_poses_sorted), max(all_mics_poses_sorted), 500)
            plt.plot(x_range, func(x_range, *popt), label=f'{round(rps,2)}')
            plt.plot(all_mics_poses_sorted, all_compensted_sigs, label=f'{round(rps,2)}, fitted', marker='o')
            powers.append(popt)
            covs.append(pcov)
        
        title_text += ', curve fitted to: `a/x**b`'
        continue

    if apply_fit:
        popt, pcov = curve_fit(func, all_mics_poses_sorted, all_compensted_sigs,
                        bounds=([0,1.5], [np.inf,4]))
                   
        x_range = np.linspace(min(all_mics_poses_sorted), max(all_mics_poses_sorted), 500)
        # x_range = np.linspace(min(all_mics_poses_sorted)-5, max(all_mics_poses_sorted)+5, 500)
        plt.plot(x_range, func(x_range, *popt), label=f'{round(rps,2)}')
        title_text += ', curve fitted to: `a/x**b`'
        powers.append(popt)
        covs.append(pcov)
    else:
        plt.plot(all_mics_poses_sorted, all_compensted_sigs, label=f'{round(rps,2)}', marker='o')
    

print(powers)
plt.title(title_text)
plt.xlabel('Mic distance from rotor\'s origin[cm]')
plt.ylabel(ylabel_text)
plt.grid()
#plt.ylim(top=0.15, bottom=0)
#plt.ylim(bottom=0)
plt.legend(loc='upper center', ncol=3, handleheight=2., labelspacing=0.002, handlelength=1.5)
plt.tight_layout()
print(f'average power: {np.average(np.array(powers),0)}')
plt.show()
plt.clf()

# %%
stds = []
for cov in covs:
    std_one = np.sqrt(np.diag(cov))
    stds.append(std_one)

print(f'Average std on optimization params: {np.average(stds, 0)}')
# only a: Average std on optimization params: [5.75441949]
# with b: Average std on optimization params: [2.66404009e+03 1.61456198e-01]

# %%
print(f'Averaged coeffs: {np.average(np.array(powers),0)}')
# only a: Averaged coeffs: [93.07030795]
# with b: Averaged coeffs: a=4.16095174e+03, b=2.93239203e+00

