import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..', '..')
sys.path.insert(0, parent_dir)

import torch
from torch.nn import MSELoss, L1Loss
#from EARS.forward_model.forward_model import get_opt_params

import numpy as np
#from EARS.pyroomacoustics_differential import forward_model_2D_interface
import EARS.forward_net.multibatch_forward_2d_interface as forward_model_2D_interface
import matplotlib.pyplot as plt
def get_opt_params(opt_params, idx_block, given_radiuses_circles=None):
    '''
    Get opt params by the correct foramt.
    '''
    return given_radiuses_circles, opt_params[:,:idx_block], opt_params[:,idx_block:]
def create_opt_params(radiuses_circles, phi_init_0, magnitudes):
    '''
    Create opt params in the correct format.
    torch version.
    '''
    return torch.concatenate([phi_init_0, magnitudes],axis=1), radiuses_circles #axis should be 1, otherwise the cat is on batch dim

def forward_func(opt_params, rir, delay_sources, num_mics, max_rec_len,
                    omega, phies0, real_recordings,
                    num_sources, fs, duration, opt_harmonies=[1], phase_shift=[0], flip_rotation_direction=[0],
                    given_radiuses_circles=None, compare_real=True, 
                    return_sim_signals=False, num_rotors=1, modulate_phase=False, recordings_foreach_rotor=False,device = 'cpu',use_mse=True,
                    use_multi_distance=False, use_all_distances=False, plot=False, path=None, exp_name=None, epoch:int=None,
                    use_fourier=False, factor=None, use_tv=False, use_lipschitz=False, use_rescaled=False):
    '''
    Computing the forward model - torch version.
    In this function the optimization params are set to the correct format,
    the signals are created and simulated (convolved) with the given RIR,
    and the loss is computed.
    Return value is: 
    - the simulated recordings by the mics (setting compare_real to False)
    - the loss (setting return_sim_signals to True)
    - both (setting compare_real and return_sim_signals to False)
    '''
    radiuses_circles, phi_init_0, magnitudes = get_opt_params(opt_params, num_sources, given_radiuses_circles)
    if num_rotors > 1:
        # phies0 = (jnp.tile(phies0, (num_rotors,1)) + jnp.expand_dims(phase_shift, 1)).reshape(num_sources*num_rotors)
        # phi_init_0 = jnp.tile(phi_init_0, (num_rotors, 1, 1))
        # magnitudes = jnp.tile(magnitudes, (num_rotors,1,1))
        phies0 = (torch.tile(phies0, (num_rotors,1)) + torch.unsqueeze(phase_shift, 1)).reshape(num_sources*num_rotors)
        phi_init_0 = torch.tile(phi_init_0, (num_rotors, 1, 1))
        magnitudes = torch.tile(magnitudes, (num_rotors,1,1))
    else:
        phies0 += phase_shift[0]
    # phies0 = np.array(phies0.reshape(phies0.shape[0], 1, 1) + phi_init_0)
    # phies0 shape: (batch_size, num_sources, num_radiuses, len_opt_harmonies)
    phies0 = phies0.reshape(phies0.shape[0], 1, 1) + phi_init_0
    #print(f"inside forward_func: phies0.shape: {phies0.shape}")
    #phies0,magnitudes = torch.as_tensor(phies0,requires_grad=True,device=device), torch.as_tensor(np.array(magnitudes),requires_grad=True,device=device)

    out = forward_model_2D_interface.create_signals_and_simulate_recordings(rir, num_mics, 
                                            fs, duration, omega, phies0, magnitudes, opt_harmonies, 
                                            num_sources_in_circle=num_sources, 
                                            radiuses_circles=radiuses_circles, delay_sources=delay_sources, flip_rotation_direction=flip_rotation_direction,
                                            max_rec_len=max_rec_len, modulate_phase=modulate_phase, recordings_foreach_rotor=recordings_foreach_rotor,
                                            use_multi_distance=use_multi_distance, use_all_distances=use_all_distances)
    if use_all_distances:
        simulated_recordings = out[0]
        premix_signals = out[1]
    else:
        simulated_recordings = out
    if not compare_real:
        return simulated_recordings

    max_len = min(real_recordings.shape[-1],simulated_recordings.shape[-1])
    if not use_mse:
        loss = L1Loss()(real_recordings[...,:max_len], simulated_recordings[...,:max_len])
    else:
        if use_fourier:
            # compute fourier transform of both signal 
            real_fft = torch.abs(torch.fft.rfft(real_recordings[...,:max_len]))
            sim_fft = torch.abs(torch.fft.rfft(simulated_recordings[...,:max_len]))
            if factor is None:
                loss = MSELoss()(real_fft,sim_fft)
            else:
                loss = MSELoss()(real_recordings[...,:max_len], simulated_recordings[...,:max_len])+\
                    factor*MSELoss()(real_fft, sim_fft)
        elif use_tv:
            if factor is None:
                raise ValueError("Please choose a factor to balance between the MSE and the TV loss")
            tv_loss = torch.mean(torch.sum(torch.abs(simulated_recordings[:,1:]-simulated_recordings[:,:-1]), dim=1))
            loss = MSELoss()(real_recordings[...,:max_len], simulated_recordings[...,:max_len])+factor*tv_loss
        elif use_lipschitz:
            if factor is None:
                raise ValueError("Please choose a factor to balance between the MSE and the Lipschitz loss")
            # use this function to show any performance warning (as indicated in the documentation for the function torch.autograd.grad)
            #torch._C._debug_only_display_vmap_fallback_warnings(True)
            # compute the gradient of the simulated recording
            abs_grad = torch.abs(torch.autograd.grad(simulated_recordings.sum(), simulated_recordings, create_graph=True, retain_graph=True)[0])
            # calculate the Lipschitz constant for each signal in the batch
            soft_lipschitz = torch.sum(torch.nn.Softmax(dim=1)(abs_grad)*abs_grad)
            loss = MSELoss()(real_recordings[...,:max_len], simulated_recordings[...,:max_len])+factor*soft_lipschitz
        elif use_rescaled:
            if factor is None:
                raise ValueError("Please choose a factor to balance between the MSE and the rescaled MSE")
            # we normalize the signals so that both of them lie in the range [0,1]
            rescaled_real = (real_recordings[...,:max_len]-real_recordings[...,:max_len].min())/(real_recordings[...,:max_len].max()-real_recordings[...,:max_len].min())
            rescaled_sim = (simulated_recordings[...,:max_len]-simulated_recordings[...,:max_len].min())/(simulated_recordings[...,:max_len].max()-simulated_recordings[...,:max_len].min())
            loss = MSELoss()(real_recordings[...,:max_len], simulated_recordings[...,:max_len])+ factor*MSELoss()(rescaled_real, rescaled_sim)
        else:
            # by using ... we are always slicing over the last dimension, no matter how many dimensions we have
            # This is important because the last dimension is the time dimension, and we want to slice over it
            # Indeed when we use multi_distance we have 3 dimensions (batch_size, num_mics, time)
            # when we don't use multi_distance we have 2 dimensions (batch_size, time)
            loss = MSELoss()(real_recordings[...,:max_len], simulated_recordings[...,:max_len])
        

    if plot:
        assert path is not None and exp_name is not None and epoch is not None and num_mics in [8,8*64, 8*80]
        plot_waveforms(real_recordings[0,...,:max_len].detach().cpu().numpy(), 
                  simulated_recordings[0,...,:max_len].detach().cpu().numpy(),
                  num_mics,path,exp_name, epoch)

    if return_sim_signals:
        return loss, simulated_recordings
    if use_all_distances:
        return loss, premix_signals
    return loss
def plot_waveforms(real, sim, num_mics_pressure_field, path, exp_name, epoch):
    '''path, exp_name and epoch are required to save the plot in the right folder with a reasonable name'''
    assert real.shape == sim.shape
    print('Starting to plot the waveforms')
    #print(f'real {real.shape} sim {sim.shape} num_mics')
    #print(f'num_mics {num_mics_pressure_field}')
    #real = np.load(os.path.join(self.config.recordings_and_encoders_dir, 'real_rec.npy'))
    #sim = np.load( os.path.join(self.config.recordings_and_encoders_dir, 'simulated_rec.npy'))
    vel = 23.46668440418565
    fs = 3003.735603735763
    mics_range = [53, 57, 63, 68, 73, 83, 93, 103]
    max_all = max(real[0].max(), sim[0].max())
    min_all = min(real[0].min(), sim[0].min())

    #subplots_labels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$']
    subplots_labels = [r'$0$', r'$\pi$ \ 2', r'$\pi$', r'$3\pi$ \ 2']

    # %%

    fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True, figsize=(7, 7))
    plt.subplots_adjust(hspace = 0.3, wspace = 0.1, left=0.15)
    #total_error = np.zeros((4, 8))
    #step_phase = int(num_mics_pressure_field/4)
    #step_distance = num_mics_pressure_field
    #print(f'step_phase: {step_phase} step_distance: {step_distance}')
    
    # if we use only one phase 
    if real.shape[0] == 8:
        for j in range(8):
            ax = axes[j//2, j%2]

            cur_real = real[j][450:740]
            ax.plot(cur_real, label=f'real', lw=0.8)

            cur_sim = sim[j][450:740]
            ax.plot(cur_sim, label=f'sim', lw=0.8)

            ax.set_title(f'Distance {mics_range[j]/100} [m]', size=10)
            #total_error[i, j] = np.linalg.norm(cur_real - cur_sim)
            #print(f"total error {mics_range[j]} {subplots_labels[i]}: {np.linalg.norm(cur_real - cur_sim)}")
            if j==7:
                lgd = ax.legend(loc='lower center', bbox_to_anchor=(-0.04, -1.2),
                                ncol=4) #, fancybox=True, shadow=True)
    else:
        # check that we are using all of the phases
        assert real.shape[0] in [8*64,8*80]
        phases = real.shape[0]//80
        step_phase = int(phases/4)
        step_distance = phases
        for j in range(8):
            ax = axes[j//2, j%2]
            for i in range(4):
                # is 20 the number of samples between quarters of pi?
                # is 80 the number of samples between a distance and another one? (since it's 4 times 20 it seems to be the case)
                # Hence they use 80 virtual microphones to simulate the 8 real ones!
                #print(f'current i {i} j {j}')
                cur_real = real[i*step_phase + j*step_distance][450:740]
                ax.plot(cur_real, label=f'real {subplots_labels[i]}', lw=0.8)

                cur_sim = sim[i*step_phase + j*step_distance][450:740]
                ax.plot(cur_sim, label=f'sim {subplots_labels[i]}', lw=0.8)

                ax.set_title(f'Distance {mics_range[j]/100} [m]', size=10)
                # total_error[i, j] = np.linalg.norm(cur_real - cur_sim)
                # print(f"total error {mics_range[j]} {subplots_labels[i]}: {np.linalg.norm(cur_real - cur_sim)}")
                #print(f"shown error {mics_range[j]} {subplots_labels[i]}: {np.linalg.norm(cur_real[450:740] - cur_sim[450:740])}")
                if j==7:
                    lgd = ax.legend(loc='lower center', bbox_to_anchor=(-0.04, -1.2),
                                    ncol=4) #, fancybox=True, shadow=True)


    plt.ylim(min_all, max_all)
    # plt.legend(loc='lower left', mode="expand", ncol=4)
    ticks = [str(round(tick / fs,2)) for tick in  plt.xticks()[0] ]
    plt.xticks(plt.xticks()[0][1:-1], ticks[1:-1])

    fig.text(0.5, 0.04, 'Time [sec]', ha='center')
    fig.text(0.04, 0.5, 'Signed magnitude', va='center', rotation='vertical')


    # plt.suptitle(f'Ground truth vs. simulated recordings, distance {mics_range[j]/100} [m]')
    plt.savefig(os.path.join(path, f'{exp_name}_opt{epoch}.pdf'), bbox_inches='tight')
    # close the current figure and the current axes
    plt.clf()
    plt.cla()
    
    # Creating another plot with the entire waveform
    vel = 23.46668440418565
    fs = 3003.735603735763
    mics_range = [53, 57, 63, 68, 73, 83, 93, 103]
    max_all = max(real[0].max(), sim[0].max())
    min_all = min(real[0].min(), sim[0].min())

    #subplots_labels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$']
    subplots_labels = [r'$0$', r'$\pi$ \ 2', r'$\pi$', r'$3\pi$ \ 2']
    
    fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True, figsize=(7, 7))
    plt.subplots_adjust(hspace = 0.3, wspace = 0.1, left=0.15)
    #total_error = np.zeros((4, 8))
    #step_phase = int(num_mics_pressure_field/4)
    #step_distance = num_mics_pressure_field
    #print(f'step_phase: {step_phase} step_distance: {step_distance}')
    
    # if we use only one phase 
    if real.shape[0] == 8:
        for j in range(8):
            ax = axes[j//2, j%2]

            cur_real = real[j][450:740]
            ax.plot(cur_real, label=f'real', lw=0.8)

            cur_sim = sim[j][450:740]
            ax.plot(cur_sim, label=f'sim', lw=0.8)

            ax.set_title(f'Distance {mics_range[j]/100} [m]', size=10)
            #total_error[i, j] = np.linalg.norm(cur_real - cur_sim)
            #print(f"total error {mics_range[j]} {subplots_labels[i]}: {np.linalg.norm(cur_real - cur_sim)}")
            if j==7:
                lgd = ax.legend(loc='lower center', bbox_to_anchor=(-0.04, -1.2),
                                ncol=4) #, fancybox=True, shadow=True)
    else:
        # check that we are using all of the phases
        assert real.shape[0] in [8*64,8*80]
        phases = real.shape[0]//80
        step_phase = int(phases/4)
        step_distance = phases
        for j in range(8):
            ax = axes[j//2, j%2]
            for i in range(4):
                # is 20 the number of samples between quarters of pi?
                # is 80 the number of samples between a distance and another one? (since it's 4 times 20 it seems to be the case)
                # Hence they use 80 virtual microphones to simulate the 8 real ones!
                #print(f'current i {i} j {j}')
                cur_real = real[i*step_phase + j*step_distance]
                ax.plot(cur_real, label=f'real {subplots_labels[i]}', lw=0.8)

                cur_sim = sim[i*step_phase + j*step_distance]
                ax.plot(cur_sim, label=f'sim {subplots_labels[i]}', lw=0.8)

                ax.set_title(f'Distance {mics_range[j]/100} [m]', size=10)
                # total_error[i, j] = np.linalg.norm(cur_real - cur_sim)
                # print(f"total error {mics_range[j]} {subplots_labels[i]}: {np.linalg.norm(cur_real - cur_sim)}")
                #print(f"shown error {mics_range[j]} {subplots_labels[i]}: {np.linalg.norm(cur_real[450:740] - cur_sim[450:740])}")
                if j==7:
                    lgd = ax.legend(loc='lower center', bbox_to_anchor=(-0.04, -1.2),
                                    ncol=4) #, fancybox=True, shadow=True)


    plt.ylim(min_all, max_all)
    # plt.legend(loc='lower left', mode="expand", ncol=4)
    ticks = [str(round(tick / fs,2)) for tick in  plt.xticks()[0] ]
    plt.xticks(plt.xticks()[0][1:-1], ticks[1:-1])

    fig.text(0.5, 0.04, 'Time [sec]', ha='center')
    fig.text(0.04, 0.5, 'Signed magnitude', va='center', rotation='vertical')


    # plt.suptitle(f'Ground truth vs. simulated recordings, distance {mics_range[j]/100} [m]')
    plt.savefig(os.path.join(path, f'{exp_name}_entire_opt{epoch}.pdf'), bbox_inches='tight')
    # close the current figure and the current axes
    plt.clf()
    plt.cla() 
    print('Plotted the waveforms!')