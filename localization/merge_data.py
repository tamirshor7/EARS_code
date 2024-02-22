import os
import numpy as np


def merge_coeff(coeff):
    data_path = f'/home/tamir.shor/EARS/data/{coeff}/inverse_problem_data_wall_x_all_0_batch8/'
    file_list = [f for f in os.listdir(data_path) if f.endswith('.npy')]

    dataset = None
    min_length = float('inf')
    for fl in file_list:
        cur_array = np.load(os.path.join(data_path, fl))
        if cur_array.shape[-1] < min_length:
            min_length = cur_array.shape[-1]

        if dataset is not None:
            dataset_labels = np.expand_dims(dataset[:, :, -1], -1)
            dataset = dataset[:, :, :-1]
            dataset = dataset[:, :, :min_length - 1]
            dataset = np.concatenate((dataset, dataset_labels), axis=-1)
        label = np.expand_dims(cur_array[:, :, -1], -1)
        data = cur_array[:, :, :-1]

        # trim
        data = data[:, :, :min_length - 1]
        cur_array = np.concatenate((data, label), axis=-1)
        if dataset is None:
            dataset = cur_array
        else:
            dataset = np.concatenate((dataset, cur_array), axis=0)

    np.save(os.path.join(data_path, f"merged_{coeff}"), dataset)


for c in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]:
    merge_coeff(c)