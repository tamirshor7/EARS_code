import os
import pandas as pd
import numpy as np

root = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/mega_dataset/default_5.0_5.0_order_1_0.5_d_0.05/"
rir_files = os.listdir(root)

coordinates = [i.removesuffix('.npy') for i in rir_files if i.endswith('.npy')]
coordinates = [i.split('_') for i in coordinates]
coordinates = [[float(j) for j in i] for i in coordinates]
df = pd.DataFrame(coordinates, columns=['x', 'y', 't'])

# check the distribution of angles
number_of_angles_per_coordinate = df.groupby(['x', 'y']).count()
number_of_angles_per_coordinate = number_of_angles_per_coordinate.reset_index()
number_of_angles_per_coordinate = number_of_angles_per_coordinate.rename(columns={'t': 'number_of_angles'})
number_of_angles_per_coordinate = number_of_angles_per_coordinate.sort_values(by='number_of_angles', ascending=False)
print(number_of_angles_per_coordinate.describe())
breakpoint()