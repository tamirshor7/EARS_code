import numpy as np
import os
import argparse
from EARS.localization.physics import PLOT_DT
import re

def add_args(parser, args=None):
    parser.add_argument('-path', default=None, type=str, help='Set the path of the files to convert')
    args = parser.parse_args(args)
    return vars(args) # vars converts the return object from parser.parse_args() to a dict

def convert_file(filename):
    original = np.load(filename)
    modified = original[:,0,:]
    modified = np.sum(original, axis=0)
    pass

def build_list_of_temporal_position(T, filename='temporal_positions.npy', dt = PLOT_DT):
    # TODO: 
    # 1. What is our T?
    # 2. What is our dt?
    temporal_positions = np.arange(0,T,dt)
    np.save(filename, temporal_positions)

def valid_filename(filename):
    # Define a regular expression pattern for the desired format
    pattern = r'^\d+_\d+\.npy$'
    
    # Use re.match to check if the filename matches the pattern
    if re.match(pattern, filename):
        return True
    else:
        return False

def extract_M_N(directory):
    min_x_coordinate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    
    assert args['path'] is not None, "Please specify a path using the flag -path"
    path = args['path']

    breakpoint()
    directory = list(filter(valid_filename, os.listdir(path)))
    # sort the files so that we access them in order
    directory = sorted(directory, key= lambda filename: (int(filename.strip().split('.')[0].split('_')[0]), int(filename.strip().split('.')[0].split('_')[1])))

    #build_list_of_temporal_position()
    
    # rotor_angular_location: just return rotor_position.npy


    # for each file:
    #   - insert in the x-axis list its x coordinate
    #   - insert in the y-axis list its y coordinate

    for file in directory:
        pass

    # At the end:
    #  - sort the x-axis list; convert it to np array; save it as spatial_locations_x_axis.npy
    #  - sort the y-axis list; convert it to np array; save it as spatial_locations_y_axis.npy
    #  - 
