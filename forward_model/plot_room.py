#%%
import sys, os
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)


cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
sys.path.insert(0, parent_dir)

from pyroomacoustics_differential import room, plot_room
from pyroomacoustics_differential.forward_model_2D_interface import *

#%%
os.environ['CUDA_VISIBLE_DEVICES']= '3'
# super important flags for JAX:
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.20'

fs = 44100
room_x = 5
room_y = 5
org_x = 2.5
org_y = 2.5
num_sources = 3
angle_rad = jnp.pi/4
start_len = 0
bpf = 52
duration = 0.5
mics_R = 0.8

# create room
corners = jnp.array([[0,0], [0, room_y], [room_x, room_y], [room_x,0]]).T  # [x,y]
room2D = room.from_corners(corners, fs=fs)

# rotor params
origin2D = jnp.array([org_x, org_y])
num_sources_circles = 2
num_sources_in_circle = 2
radiuses_circles = [0.1]#[0.1,0.2]

#sources = create_circular_phased_rotor_array(origin2D, num_sources_circles, num_sources_in_circle, radiuses_circles)

sources_distances = jnp.array([-0.1, 0, 0.1])
phi_array = 0.

omega = 30

phies0 = jnp.array([-jnp.pi/2, 0, jnp.pi/2, jnp.pi,
                    -jnp.pi/2, 0, jnp.pi/2, jnp.pi,
                    0, jnp.pi/2, jnp.pi, -jnp.pi/2,
                    0, jnp.pi/2, jnp.pi, -jnp.pi/2])
                    
phies0 = jnp.zeros(len(sources_distances))
magnitudes = jnp.ones(len(sources_distances))
magnitudes = jnp.reshape(magnitudes, (magnitudes.shape[0], 1))

delays = jnp.zeros(len(sources_distances))

sources = add_sources_phased_rotor_array(origin2D,
                            omega, phies0, magnitudes, delays,
                            distances=sources_distances, phi_array=phi_array,
                            enable_prints=True)


# mics_distances = jnp.array([0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.2])
# phi_mics = jnp.array([0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5]) * jnp.pi

mics_distances = jnp.array([.7, .7, .7, .7])
phi_mics = jnp.array([0.25, 0.75, 1.25, 1.75]) * jnp.pi

mics_array = add_mics_to_room(origin2D, fs=fs, 
                                num_mics=mics_distances.shape[0], distances=mics_distances, phi_mics = phi_mics,
                                enable_prints=False)

#%%
plot_room.plot2DRoom(room2D, mics_array, sources, img_order=0, marker_size=5)
