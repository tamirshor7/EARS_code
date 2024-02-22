import os
#origins = [ (1.2, 1.7), (1.4, 3.8), (1.9, 2.1),  (2.4, 1.7), (2.5, 2.5), (2.5, 3.5), (2.8, 3.1), (3.2, 2.1), (3.6, 3.7) ]
origins = [ (1.2, 1.3), (2.4, 1.3), (2.5, 2.5)] #(2.5, 2.5), (2.5, 3.5), (2.8, 3.1), (3.2, 2.1), (3.6, 3.7) ]
exp_properties = {
                    'same_phase': {'modulate_phase':False, 'phase_0': 0, 'phase_1':0},
                    'phase_45': {'modulate_phase':False, 'phase_0': 0, 'phase_1':45},
                    'phase_minus45_45': {'modulate_phase':False, 'phase_0': -45, 'phase_1':45},
                    'phase_mod': {'modulate_phase':True, 'phase_0': 0, 'phase_1':0},
                    }

gpu=0
exp_name = 'same_phase'
exp_prop = exp_properties[exp_name]
duration=0.5
max_order = 4
indoor_modes = ['indoor'] + [f'images{x}' for x in range(max_order+1)]

# # RIR ONLY
for origin in origins:
    os.system(f'python forward_model_wrapper.py -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -max_order {max_order} -num_rotors 2 -org_x {origin[0]} -org_y {origin[1]}')


# SIMULATE
# for origin in origins:
#     for indoor_mode in indoor_modes:
        
#         if exp_prop['modulate_phase']:
#             os.system(f'python forward_model_wrapper.py -gpu {gpu} -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -max_order {max_order} -num_rotors 2 -saved_data_dir_path entire_room_{origin[0]}_{origin[1]}_{exp_name} -org_x {origin[0]} -org_y {origin[1]} -indoor_mode {indoor_mode} -phase_shift {exp_prop["phase_0"]} {exp_prop["phase_1"]} -modulate_phase')
#         else:
#             os.system(f'python forward_model_wrapper.py -gpu {gpu} -exp_name Nov25_08-28-20_aida_2_rads_80mics_128sources_051 -second_rad 0.51 -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 7 -duration 0.5 -channel 0 -max_order {max_order} -num_rotors 2 -saved_data_dir_path entire_room_{origin[0]}_{origin[1]}_{exp_name} -org_x {origin[0]} -org_y {origin[1]} -indoor_mode {indoor_mode} -phase_shift {exp_prop["phase_0"]} {exp_prop["phase_1"]}')

