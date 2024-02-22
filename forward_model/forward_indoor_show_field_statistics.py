#%% imports
import numpy as onp
import os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from pyroomacoustics_differential import consts
from forward_indoor_wrapper import correlate_each_channel

#%% helper funcs

def shift_recordings_min(recordings, phase_shifts, window_len=128, start_revolution=5, num_revolutions=5):
    '''here after aggregating each recording by summing over the rotors we take the minimum of the recordings over the revolutions
    Why are we taking the minimum?'''
    all_recordings = []
    for i in range(num_revolutions):
        cur_recordings = recordings[:, :, window_len * (start_revolution+i):window_len * (start_revolution+i+1)]
    
        
        # run over the rotors (recordings.shape[0]) and change phase according to phase_shifts
        for i in range(recordings.shape[0]):
            samples_shift = int((phase_shifts[i] / 360) * window_len)
            cur_recordings[i] = onp.roll(cur_recordings[i], samples_shift, axis=-1)
        
        cur_recordings = onp.sum(cur_recordings, axis=0)
        all_recordings.append(cur_recordings)
    all_recordings_min = onp.min(onp.array(all_recordings), axis=0)
    return all_recordings_min

def shift_recordings(recordings, phase_shifts, window_len=128, start_revolution=5, num_revolutions=5):
    '''shifts the recordings according to the phase shifts of the rotors'''
    cur_recordings = recordings[:, :, int(window_len * start_revolution):int(window_len * (start_revolution+num_revolutions))]
    # run over the rotors (recordings.shape[0]) and change phase according to phase_shifts
    for i in range(recordings.shape[0]):
        # here we are shifting the samples in the time axis proportionally to the phase shift (in particular we first normalize the shift byh dividing it by 360
        # and then we multiply it by the window length)
        samples_shift = int((phase_shifts[i] / 360) * window_len)
        cur_recordings[i] = onp.roll(cur_recordings[i], samples_shift, axis=-1)
    # sum rotors' inputs
    # I guess that we are modelling recordings as the superposition (in this case a simple sum) of the recordings of all of the rotors
    cur_recordings = onp.sum(cur_recordings, axis=0)
    return cur_recordings


def read_recordings(x, y, dir_recording_path, phase_shifts, start_revolution, num_revolutions):
    recordings_path = os.path.join(dir_recording_path, f'{int(x*100)}_{int(y*100)}.npy')
    # originally recordings shape: (rotor, channel, samples) (I still have some doubts about the last two axis)
    recordings = shift_recordings(onp.load(recordings_path), phase_shifts, start_revolution=start_revolution, num_revolutions=num_revolutions)
    # after shift recordings shape: (channel, samples) (I still have some doubts about these dimensions)
    return recordings

# for each location on the grid compute correlation to a specific position
def compute_grid(dir_recording_path, org_id, phase_shifts, plot_grid=True, thresh=None, start_revolution=5, num_revolutions=5):
    # compute org recordings
    org = (xx[org_id[0]], yy[org_id[1]])
    org_recordings = read_recordings(org[0], org[1], dir_recording_path, phase_shifts, start_revolution, num_revolutions=num_revolutions)

    #compute grid
    grid = onp.zeros((len(xx), len(yy)))
    for id_x, x in enumerate(xx):
        for id_y, y in enumerate(yy):

            recordings = read_recordings(x, y, dir_recording_path, phase_shifts, start_revolution, num_revolutions=num_revolutions)
            corr = correlate_each_channel(org_recordings, recordings)
            grid[id_x, id_y] = corr

    if plot_grid:
        fig,ax = plt.subplots()
        ax.set_title(f'Phase shifts by rotor: {phase_shifts} [deg]')
        img = ax.imshow(grid)
        
        if thresh is not None:
            ax.contour(grid, [thresh], origin='lower', cmap='flag', extend='both',
                            linewidths=0.5)

        ax.scatter(org_id[1], org_id[0], color='red', marker='x', s=16, linewidths=1)
        fig.colorbar(img)
        # plt.axis('off')
        x_ticks_vals = plt.xticks()[0]
        x_labels = [round(xx[int(x_ticks_vals[i])],2) for i in range(0, len(x_ticks_vals)-1)]
        ax.set_xticklabels(x_labels)

        y_ticks_vals = plt.yticks()[0]
        y_labels = [round(yy[int(y_ticks_vals[i])],2) for i in range(0, len(y_ticks_vals)-1)]
        ax.set_yticklabels(y_labels)
        plt.show()

    return grid

def compute_grid_thresh_area(grid, thresh, delta):
    entry_area = delta**2 # meter
    grid[grid >= thresh] = 1
    area = len(onp.where(grid == 1)[0]) * entry_area
    return area

def compute_bb_around_all_peaks(grid, thresh):
    above_thresh_positions = onp.where(grid>=thresh)
    max_x = above_thresh_positions[0].max()
    max_y = above_thresh_positions[1].max()
    min_x = above_thresh_positions[0].min()
    min_y = above_thresh_positions[1].min()
    return min_x, max_x, min_y, max_y

#%% arguments and parameters definitions
num_rotors = 4
start_revolution = 5
room_x = 5.
room_y = 5.
rotor_type = 18
delta = 0.05
margins = 0.05
thresh = 0.75
phase_shift_indexing = onp.arange(0,360,180)
org_id = (55, 58) #(1,50) (25,40) (40, 23), (55, 58)

# phase_shifts = [[ 0, 0, 0, 0]]
# dir_recording_path = f'indoor_recordings_{num_rotors}_rotors_4_mics_d_{delta}_mod_[0, 1, 1, 0]'
# phase_shifts = [[ 0, 0, 0, 0], [ 0, 45, 45, 0],  [0, 90, 90, 0]]
# dir_recording_path = f'indoor_recordings_4_rotors_4_mics_d_0.05_0110'


recordings_info = [
                    {'dir_recording_path':'indoor_recordings_4_rotors_4_mics_d_0.05_0110', 'phase_shifts': [[ 0, 0, 0, 0],  [0, 45, 45, 0],  [0, 90, 90, 0]],
                                                                                            'titles': ['(0,0,0,0)', r'(0,$\pi/4$,$\pi/4$,0)', r'(0,$\pi/2$,$\pi/2$,0)']},
                    {'dir_recording_path': f'indoor_recordings_4_rotors_4_mics_d_0.05_mod_[0, 1, 1, 0]', 'phase_shifts': [[ 0, 0, 0, 0]]}]

rotor_length_meter = consts.inch_to_meter(rotor_type)
mics_R = 2*rotor_length_meter

xx = onp.arange( round((mics_R+margins), 2), round((room_x-mics_R-margins),2), delta)
yy = onp.arange( round((mics_R+margins), 2), round((room_y-mics_R-margins),2), delta)

#%% compute grid for each phase shift and org_id
grids = []
     
#for org_id in orgs_ids:

recording = recordings_info[0]
for phase_shift in recording['phase_shifts']:
    cur_grid = compute_grid(recording['dir_recording_path'], org_id, phase_shift, plot_grid=False, thresh=thresh, start_revolution=start_revolution, num_revolutions=1)
    area = compute_grid_thresh_area(cur_grid, thresh, delta)
    grids.append({'grid': cur_grid.copy(), 'area': round(area,2)})
    #bb = compute_bb_around_all_peaks(cur_grid, thresh)

# compute min grid and add it to the grids list
min_grid = onp.min([grid['grid'] for grid in grids], axis=0)
area = compute_grid_thresh_area(min_grid, thresh, delta)
grids.append({'grid': min_grid.copy(), 'area': round(area,2)})

# compute phase modulations
# recording = recordings_info[1]
# phase_shift = recording['phase_shifts'][0]

# phase_mod_grid = compute_grid(recording['dir_recording_path'], org_id, phase_shift, plot_grid=False, thresh=thresh, start_revolution=5, num_revolutions=5)
# area = round(compute_grid_thresh_area(phase_mod_grid, thresh, delta),2)
# grids.append({'grid': phase_mod_grid.copy(), 'area': round(area,2)})

# #%%
# fig, axes = plt.subplots(2,3,
#                         gridspec_kw=dict(width_ratios=[2,2,2.5]))
# plt.subplots_adjust(hspace=0.15, wspace=0.3, left=0.13)
# for i in range(5):

#     cur_grid = grids[i]

#     if i < 3:
#         ax = axes[i//2,i%2]
#         ax.set_title(f'{recordings_info[0]["titles"][i]}, area {cur_grid["area"]}', size=10)
#     elif i == 3:
#         ax = axes[0,2]
#         ax.set_title(f'Minimum, area {cur_grid["area"]}', size=10)
#     else:
#         ax = axes[1,2]
#         ax.set_title(f'Modulations, area {cur_grid["area"]}', size=10)
    
    
#     img = ax.imshow(cur_grid['grid'], vmin=-1, vmax=1)
    
#     if thresh is not None:
#         ax.contour(cur_grid['grid'], [thresh], origin='lower', cmap='flag', extend='both',
#                         linewidths=0.5)

#     ax.scatter(org_id[1], org_id[0], color='red', marker='x', s=16, linewidths=1)
    
    
#     if i == 2:
#         x_ticks_vals = ax.get_xticks()[1:-1]
#         y_ticks_vals = ax.get_yticks()[1:-1]

#         ax.set_xlabel('Distance [m]', size=8)
#         ax.set_ylabel('Distance [m]', size=8)
        
#         x_labels = [0] +[round( xx[int(tick)],2) for tick in x_ticks_vals]
#         y_labels = [0] + [round( yy[int(tick)],2) for tick in y_ticks_vals]
#         ax.set_xticklabels(x_labels, size=8)
#         ax.set_yticklabels(y_labels, size=8)
    
#     else:
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])

# fig.subplots_adjust(left=0.1, right=0.9)
# cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7]) #  [left, bottom, width, height]
# fig.colorbar(img, cax=cbar_ax)

# plt.show()
# plt.clf()


# %%
def final_plot(grids, hspace=-0.4, wspace=0.5, x_max=3.3, modulation='minimum', vlines=[1, 1.15, 2.15,2.3]):
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(6, 8, hspace=hspace, wspace=wspace)

    ax = fig.add_subplot(gs[4, :6])

    xl = 3600
    phases = [0, onp.pi/4, onp.pi/2]
    if modulation=='minimum':
        phases_revolutions = onp.ones(xl) * 0.2

        full_revolution = 1100
        delta = 150
        for i in range(3):
            phases_revolutions[full_revolution*i + delta*i:full_revolution*(i+1) + delta*i] = phases[i]
            
            if i!= 2:
                phases_revolutions[full_revolution*(i+1)+delta*i:full_revolution*(i+1)+delta*(i+1)] = onp.linspace(phases[i], phases[i+1], delta)
        ax.plot(onp.linspace(0,x_max,xl), phases_revolutions)

    elif modulation=='linear':
        ax.plot([0,x_max], [phases[0], phases[-1]] )
    
    ax.vlines(x=vlines, color='k',linestyle=(0, (5, 7)), ymin=0, ymax=onp.pi/2, lw=0.8)

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlim([0, x_max])

    ax.set_xlabel(f'Modulation:\n{modulation}', size=12)
    ax.set_ylabel('Rotor Phase [rad]', size=8)
    ax.set_xlabel('Revolution', size=8)


    for i in range(3):
        cur_grid = grids[i]
        ax = fig.add_subplot(gs[:4, i*2:i*2+2])
        # ax.plot(onp.arange(1., 0., -0.1) * 2000., onp.arange(1., 0., -0.1))
        ax.imshow(cur_grid['grid'], vmin=-1, vmax=1)
        
        if thresh is not None:
            ax.contour(cur_grid['grid'], [thresh], origin='lower', cmap='flag', extend='both',
                            linewidths=0.5)

        ax.scatter(org_id[1], org_id[0], color='red', marker='x', s=16, linewidths=1)
        
        ax.set_title(f'Area {cur_grid["area"]}', size=8)
        if i == 0:
            x_ticks_vals = ax.get_xticks()[1:-1]
            y_ticks_vals = ax.get_yticks()[1:-1]

            # ax.set_xlabel('Distance [m]', size=8)
            ax.set_ylabel('Distance [m]', size=8)
            
            x_labels = [0] +[round( xx[int(tick)],2) for tick in x_ticks_vals]
            y_labels = [0] + [round( yy[int(tick)],2) for tick in y_ticks_vals]
            ax.set_xticklabels(x_labels, size=8)
            ax.set_yticklabels(y_labels, size=8)
        
        else:
            x_labels = [0] +[round( xx[int(tick)],2) for tick in x_ticks_vals]
            ax.set_xticklabels(x_labels, size=8)
            ax.set_yticklabels([])
            ax.set_yticks([])


    cur_grid = grids[3]
    ax = fig.add_subplot(gs[2:5, 6:])
    # ax.plot(onp.arange(1., 0., -0.1) * 2000., onp.arange(1., 0., -0.1))
    img=ax.imshow(cur_grid['grid'], vmin=-1, vmax=1)

    if thresh is not None:
        ax.contour(cur_grid['grid'], [thresh], origin='lower', cmap='flag', extend='both',
                        linewidths=0.5)

    ax.scatter(org_id[1], org_id[0], color='red', marker='x', s=16, linewidths=1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(f'Modulation: {modulation}\nArea {cur_grid["area"]}', size=8)
    x_labels = [0] +[round( xx[int(tick)],2) for tick in x_ticks_vals]
    ax.set_xticklabels(x_labels, size=8)

    ax.set_yticklabels([])

    fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()

    fig.subplots_adjust(left=0.1, right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.6]) #  [left, bottom, width, height]
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8) 


    plt.tight_layout()
    plt.savefig(f'modulation_localization_{org_id}_{modulation}.pdf')
    plt.clf()

# %%
final_plot(grids)#, wspace=0.05)


#%%
# compute phase modulations
grids = []
recording = recordings_info[1]
phase_shift = recording['phase_shifts'][0]

# revolutions=[5,7,9]
vlines = [1.5,1.75,3.3,3.55]
start_revolutions = [5, 6.75, 8.55]
for i in range(3):
    cur_grid = compute_grid(recording['dir_recording_path'], org_id, phase_shift, plot_grid=False, thresh=thresh, start_revolution=start_revolutions[i], num_revolutions=vlines[0])
    area = compute_grid_thresh_area(cur_grid, thresh, delta)
    grids.append({'grid': cur_grid.copy(), 'area': round(area,2)})
    #bb = compute_bb_around_all_peaks(cur_grid, thresh)

# compute min grid and add it to the grids list
mod_grid = compute_grid(recording['dir_recording_path'], org_id, phase_shift, plot_grid=False, thresh=thresh, start_revolution=start_revolution, num_revolutions=5)
area = compute_grid_thresh_area(mod_grid, thresh, delta)
grids.append({'grid': mod_grid.copy(), 'area': round(area,2)})

final_plot(grids, x_max=5, modulation='linear', vlines=[1.5,1.75,3.3,3.55])
# %%
