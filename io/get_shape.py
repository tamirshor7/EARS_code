import h5py
import os
from tqdm import tqdm
import numpy as np
from EARS.io.fast_io import get_listed_files
import pandas as pd

#path = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/orig_64_angles/default_5.0_5.0_order_1_0.5_d_0.05/"
path = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/non_convex_room/non_convex_5.0_5.0_order_1_0.5_d_0.05/"
shapes = []
for i in tqdm(get_listed_files(path)):
    if not i.endswith(".hdf5"):
        continue
    try:
        with h5py.File(os.path.join(path, i), "r") as f:
            # print(f"shape: {f['data'].shape}")
            # exit()
            shapes.append(f['data'].shape)
    except:
        print(f"{i} is corrupted. Renaming it...")
        old_name = os.path.join(path, i)
        new_name = old_name.removesuffix(".hdf5")
        new_name = f"{new_name}.corrupted"
        os.rename(old_name, new_name)

df = pd.DataFrame(shapes, columns=["rotors", "microphones", "time"])
df.to_csv("shapes.csv")
print(f"max is {df.max()}")
shapes = np.array(shapes)
print(shapes)
m = np.max(shapes, axis=0)
print(f"max is {m}")