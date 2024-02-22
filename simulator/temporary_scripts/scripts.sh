python EARS/simulator/temporary_scripts/elaborate.py --mode "free_space" \
 --path "entire_room_org_2.5_2.5_rotors_1_free_space" 

python EARS/simulator/temporary_scripts/elaborate.py --mode "free_space" \
 --path "entire_room_org_2.5_2.5_rotors_4_free_space_and_indoor" 

python EARS/simulator/temporary_scripts/elaborate.py --mode "indoor" \
 --path "entire_room_org_2.5_2.5_rotors_4_free_space_and_indoor" 

python EARS/simulator/temporary_scripts/elaborate_modulate_phase.py --mode "indoor" \
 --path "padded_entire_room_org_2.5_2.5_rotors_4_free_space_and_indoor" \
 --phase_modulation "constant_offset"

python EARS/simulator/temporary_scripts/elaborate_modulate_phase.py --mode "indoor" \
 --path "padded_entire_room_org_2.5_2.5_rotors_4_free_space_and_indoor" \
 --phase_modulation "time_varying_sine"

python EARS/simulator/temporary_scripts/elaborate_modulate_phase.py --mode "free_space" \
 --path "entire_room_org_1.5_1.0_rotors_4_free_space_and_indoor" \
 --phase_modulation "time_varying_sine" \
 --org_x 1.5 --org_y 1.0

python EARS/simulator/temporary_scripts/elaborate_modulate_phase.py --mode "indoor" \
 --path "entire_room_org_1.5_1.0_rotors_4_free_space_and_indoor" \
 --phase_modulation "time_varying_sine" \
 --org_x 1.5 --org_y 1.0

python EARS/simulator/temporary_scripts/elaborate_modulate_phase.py --mode "indoor" \
 --path "padded_entire_room_org_2.5_2.5_rotors_4_free_space_and_indoor" \
 --phase_modulation "/mnt/walkure_public/tamirs/phase_modulations/learned_phase_modulation.npy"