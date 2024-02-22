import numpy as np
import pandas as pd
import os

def filter_additional_rows(group):
    pass

root = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/default_5.0_5.0_order_1_0.5_d_0.05/rir/rir_indoor/"
rirs = os.listdir(root)
sound_path = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/default_5.0_5.0_order_1_0.5_d_0.05/"
sounds = os.listdir(sound_path)


coordinates = [i.removesuffix('.npy') for i in rirs if not i.endswith('.corrupted')]
coordinates = [tuple(i.split('_')) for i in coordinates]
coordinates = [(float(i[0]),float(i[1]),float(i[2])) for i in coordinates]

df = pd.DataFrame(coordinates, columns=['x','y','t'])
counter = df.groupby(['x','y']).count()
number_of_values_to_remove = counter['t']-16
values_to_subsample_from = df.groupby(['x','y']).filter(lambda group: ~group.isin([0, np.pi/2, np.pi, 3*np.pi/2]))

sound_coordinates = [i.removesuffix('.npy') for i in sounds if not i.endswith('.corrupted')]
sound_coordinates = [tuple(i.split('_')) for i in sound_coordinates]
sound_coordinates = [(float(i[0]),float(i[1]),float(i[2])) for i in sound_coordinates]


# for each x,y pair in counter:
#   number of values to remove = counter[x,y]-16
#   candidate values = all of the rows of x,y \ [0,pi/2,pi, 3pi/2]
#   add these candidate values to removal dataframe
# for each row in removal dataframe
#   convert the coordinates to the name of the file
#   remove that file







# -----------------------------------------------------------
# cancel all of the corrupted files
# construct a dataframe containing all of the coordinates
# for each over sampled point choose uniformly random among the list of their angles (besides 0,pi/2,3pi/2) to drop
#   cancel them
# construct a dataframe containing for each subsampled point the amount of random coordinates needed to reach 32 per point
#   compute these random angles and put them in a dataframe along with their coordinate
# construct a list out of this
# divide these elements of the list among all of the processes