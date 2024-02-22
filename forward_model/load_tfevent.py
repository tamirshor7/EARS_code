#%%
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os, sys
import numpy as np
import matplotlib.pyplot as plt

cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)
from data_processing.create_signals_manually import *

class TFeventParams:
    storage_location = 'walkure_public' # 'walkure_public'
    def __init__(self, exp_name, omega, num_sources_power, opt_harmonies, 
            is_opt_radius=False, 
            given_radiuses_circles=-1,
            fs=44100, duration=0.6, exp_dir=f'/mnt/{storage_location}/tamirs/EARS/runs', which_step=-1):
                                                                
        # forward unoptimized params
        self.duration = duration
        self.omega = omega
        self.fs = fs
        self.harmonies = opt_harmonies
        self.num_sources = 2**num_sources_power
        # load tfevent file with event accumulator
        event_acc = EventAccumulator(os.path.join(exp_dir, exp_name))
        event_acc.Reload()
        # print(event_acc.Tags()) # print all TBoard tags
        
        # get loss val
        #_, _, vals = zip(*event_acc.Scalars('Loss'))
        vals = list(map(lambda x: x.value, event_acc.Scalars('Loss')))
        self.loss = np.asarray(vals)

        radiuses_num = len(set([int(x.split('/')[2].split('_')[1]) for x in event_acc.Tags()['scalars'] if x.startswith('Signals/Phies/Radius')]))
        self.harmonies_coeffs = np.empty((self.num_sources, radiuses_num, len(self.harmonies), self.loss.shape[0]))
        self.phies_0 = np.empty_like(self.harmonies_coeffs)

        # get radius
        if is_opt_radius:
            self.radiuses_circles = np.empty((radiuses_num, self.loss.shape[0]))
            for i in range(2*self.num_sources, 2*self.num_sources+radiuses_num):
                _, _, vals = zip(*event_acc.Scalars(f'Signals/Radius_{i}'))
                self.radiuses_circles[i] = vals
        else:
            self.radiuses_circles = given_radiuses_circles

        # get harmonies and phi0
        for rad_i in range(radiuses_num):
            for source_i in range(self.num_sources):
                for harmony_i, harmony in enumerate(opt_harmonies):
                    # get optimized harmony coeffs
                    harmony = round(float(harmony),1)
                    #_, _, vals = zip(*event_acc.Scalars(f'Signals/Magnitudes/Radius_{rad_i}/Source_{source_i}/m_i_{harmony}'))
                    vals = list(map(lambda x: x.value, event_acc.Scalars(f'Signals/Magnitudes/Radius_{rad_i}/Source_{source_i}/m_i_{harmony}')))
                    self.harmonies_coeffs[source_i, rad_i, harmony_i] = vals
                    # get optimized phi0
                    #_, _, vals = zip(*event_acc.Scalars(f'Signals/Phies/Radius_{rad_i}/Source_{source_i}/p_i_{harmony}'))
                    vals = list(map(lambda x: x.value, event_acc.Scalars(f'Signals/Phies/Radius_{rad_i}/Source_{source_i}/p_i_{harmony}')))
                    self.phies_0[source_i, rad_i, harmony_i] = vals
        self.which_step = which_step


    def get_params(self):
        # get the last iteration optimized params
        loss = self.loss[self.which_step]

        if type(self.radiuses_circles) is float or type(self.radiuses_circles) is np.float64:
            radiuses_circles = self.radiuses_circles
        else:
            radiuses_circles = self.radiuses_circles

        optimized_harmonies_coeffs = self.harmonies_coeffs[:,:,:,self.which_step]
        optimized_phies_0 = self.phies_0[:,:,:,self.which_step]
        
        return loss, radiuses_circles, optimized_phies_0, optimized_harmonies_coeffs, self.harmonies


    def plot_emitted_signal(self):
        _, _, optimized_phies_0, optimized_harmonies_coeffs, _ = self.get_params()
        sample = create_harmonic_signal_matrix_style_with_coeff(self.omega, optimized_harmonies_coeffs, phi_0=optimized_phies_0, 
                                                                harmonies=self.harmonies, fs=self.fs, duration=self.duration)
        plt.plot(sample)
        plt.show()
        plt.clf()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']= '3'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.20'
    # This experiment has been deleted! It can't be loaded!
    exp_name = 'Oct25_15-42-53_aida_two_rads_dif_dist_two_org_y'
    
    tf_event_params = TFeventParams(exp_name=exp_name, omega=23, fs=44100, duration=1.,
                                    num_sources_power=4,
                                    opt_harmonies=[0.5,1,2,3], given_radiuses_circles=[3])
    loss, radiuses_circles, optimized_phi_0, optimized_harmonies_coeffs, harmonies = tf_event_params.get_params()
    #print(f'Loss: {loss}\nOptimized phi_0: {optimized_phi_0}\nOptimized harmonies coeffs: {optimized_harmonies_coeffs}')
    # tf_event_params.plot_optimization_process(plot_harmonies=True)
# %%
