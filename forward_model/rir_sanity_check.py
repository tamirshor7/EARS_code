import os
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
direct_component_path = os.path.join(cur_dir, '..', '..', 'data', 'rir', 'direct_component', 'direct_component.npy')
direct_component = np.load(direct_component_path)
print(f"Direct component shape: {direct_component.shape}")

differences = []
first_order_rir_list = []

first_order_path = os.path.join(cur_dir, '..', '..', 'data', 'rir', 'first_order_0.5')
complete_rir_path = os.path.join(cur_dir, '..', '..', 'data', 'rir', 'rir_indoor_4_channels_5.0_5.0_order_1_0.5_d_0.05','rir_indoor')
intersection = set(os.listdir(first_order_path)).intersection(set(os.listdir(complete_rir_path)))
print(f"Matching files: {len(intersection)}/{len(os.listdir(first_order_path))}")

for file in intersection:
    complete_rir = np.load(os.path.join(complete_rir_path,file))
    first_order_rir = np.load(os.path.join(first_order_path,file))
    padding_size = ((0,0),(0,0),(0, complete_rir.shape[-1] - direct_component.shape[-1]))
    direct = np.pad(direct_component, padding_size)
    assert complete_rir.shape == direct.shape == first_order_rir.shape, f"Shape mismatch: complete {complete_rir.shape} direct {direct.shape} first {first_order_rir.shape}"
    first_order_rir_hat = complete_rir-direct
    differences.append(np.linalg.norm(first_order_rir-first_order_rir_hat))
    first_order_rir_list.append(first_order_rir)

print(f"mean difference: {np.mean(differences)} std difference: {np.std(differences)}")
average_norm = np.mean(list(map(lambda x: np.linalg.norm(x), first_order_rir_list)))
print(f"Mean norm of first order {average_norm} (ratio: {np.mean(differences)/average_norm})")