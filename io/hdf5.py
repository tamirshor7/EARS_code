import h5py
import numpy as np
import torch
import jaxlib

DATASET_NAME='data'

def load_numpy(path, dataset_name=DATASET_NAME):
    assert path.endswith(".hdf5"), f"The path must end with .hdf5, but got {path}"
    with h5py.File(path, "r") as f:
        numpy_array = np.asarray(f[dataset_name][...])
    return numpy_array

def get_shape(path, dataset_name=DATASET_NAME):
    assert path.endswith(".hdf5"), f"The path must end with .hdf5, but got {path}"
    with h5py.File(path, "r") as f:
        shape = f[dataset_name].shape
    return shape

def load_torch(path, dataset_name=DATASET_NAME):
    assert path.endswith(".hdf5"), f"The path must end with .hdf5, but got {path}"
    with h5py.File(path, "r") as f:
        torch_tensor = torch.as_tensor(f[dataset_name][...])
    return torch_tensor

def save(path, data, dataset_name=DATASET_NAME):
    assert path.endswith(".hdf5"), f"The path must end with .hdf5, but got {path}"
    
    with h5py.File(path, "w") as f:
        dset = f.create_dataset(dataset_name, data.shape, dtype='float32')
        dset[...] = data