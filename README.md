# Active propulsion noise shaping for multi-rotor aircraft localization


## Gabriele Serussi*, Tamir Shor*, Tom Hirshberg, Chaim Baskin, Alex Bronstein  -  <img src="images/vista_logo.png" width="40px" align="right"/>
## Technion Institute of Technology <img src="images/technion_logo.png" width="40px" align="right"/>
\* Denotes equal contribution  
>
> Multi-rotor aerial autonomous vehicles (MAVs)
primarily rely on vision for navigation purposes. However,
visual localization and odometry techniques suffer from poor
performance in low or direct sunlight, a limited field of view,
and vulnerability to occlusions. Acoustic sensing can serve as
a complementary or even alternative modality for vision in
many situations, and it also has the added benefits of lower
system cost and energy footprint, which is especially important
for micro aircraft. This paper proposes actively controlling and
shaping the aircraft propulsion noise generated by the rotors to
benefit localization tasks, rather than considering it a harmful
nuisance. We present a neural network architecture for self noise-based localization in a known environment. We show
that training it simultaneously with learning time-varying rotor
phase modulation achieves accurate and robust localization.
The proposed methods are evaluated using a computationally
affordable simulation of MAV rotor noise in 2D acoustic
environments that is fitted to real recordings of rotor pressure
fields.
<br>
### Method Overview
<p align="center">
<img src="images/fwd-model-1.png" width="600px"/>  
</p>
<br>

### Tested Phase Modulation
<img src="images/IMG-20240220-WA0004.jpg" width="400px"/>

### Error spatial distribution
<img src="images/IMG-20240215-WA0025.jpg" width="200px"/> 

<br>

### Pressure Field Simulation Video:
  
[![Click to play](https://img.youtube.com/vi/pRnapzl87M8/maxresdefault.jpg)]([https://www.youtube.com/watch?v=RT3aGX-p-C0](https://youtu.be/pRnapzl87M8))



## Description  
Harness and shape the propulsion noise of multi-rotor aircraft to improve localization accuracy. We propose a system for self-noise-based localization in a known environment. We show that training it simultaneously with learning time-varying rotor phase modulation achieves accurate and robust localization. The proposed methods are evaluated using a computationally affordable simulation of MAV rotor noise in 2D acoustic environments that is fitted to real recordings of rotor pressure fields.

## Setup

### Environment
To set up our environment, please run:

```
pip install -r requirements.txt
```
or  
```
conda install --file requirements.txt

```

## Usage

### Repo Layout 
Repository layout:
```
.
├── LICENSE
├── README.md
├── _config.yml
├── controller
│   └── controller.py
├── data_processing
│   ├── create_signals_manually.py
│   └── pre_processing.py
├── forward_model
│   ├── RealRecordings.py
│   ├── forward_indoor_wrapper.py
│   ├── forward_model.py
│   ├── forward_model_wrapper.py
│   ├── jax_scipy_minimize_forward.py
│   ├── load_tfevent.py
│   ├── non_convex_room.py
│   ├── optimized_data
│   └── real_recordings
├── images
│   ├── IMG-20240215-WA0025.jpg
│   ├── IMG-20240220-WA0004.jpg
│   ├── VID-20240218-WA0003.mp4
│   ├── forward_model.png
│   ├── fwd-model-1.png
│   ├── technion_logo.png
│   └── vista_logo.png
├── io
│   ├── fast_io.py
│   └── hdf5.py
├── localization
│   ├── evaluate.py
│   ├── multi_position
│   │   ├── aggregator.py
│   │   ├── constant_speed.py
│   │   ├── dataset.py
│   │   ├── master.py
│   │   └── trajectory_factory.py
│   ├── penalty.py
│   ├── phase_modulation
│   │   ├── input_sound_parameters.npz
│   │   ├── modulation_dataset.py
│   │   ├── phase_modulation_injector.py
│   │   └── phase_modulation_pipeline.py
│   ├── physics.py
│   ├── preprocess.py
│   ├── train_pipeline.py
│   ├── train_separate.py
│   └── transformer_encoder.py
├── pyroomacoustics_differential
│   ├── acoustic_methods.py
│   ├── acoustic_methods_onp.py
│   ├── consts.py
│   ├── forward_model_2D_interface.py
│   ├── geometry.py
│   ├── geometry_onp.py
│   ├── plot_room.py
│   └── room.py
├── requirements.txt
├── scripts
│   ├── acoustic_reflection_coefficient_dataset_generation.sh
│   ├── aspect_ratio_dataset_generation.sh
│   ├── newton_cluster_point_cpu.sh
│   ├── newton_cluster_point_gpu.sh
│   ├── rir_shear_shard.sh
│   ├── shear_dataset_generation.sh
│   ├── shifted_room_cpu.sh
│   ├── shifted_room_gpu.sh
│   ├── shifted_room_newton_cpu.sh
│   ├── shifted_room_newton_gpu.sh
│   ├── shifted_room_newton_gpu_wrapper.sh
│   ├── small_dense_room.sh
│   ├── small_dense_room_gpu.sh
│   ├── train_dataset_generation_cpu.sh
│   ├── train_dataset_generation_cpu_newton.sh
│   ├── train_dataset_generation_gpu.sh
│   ├── train_dataset_generation_gpu_newton.sh
│   ├── train_dataset_generation_gpu_newton_wrapper.sh
│   ├── train_dataset_generation_gpu_shear.sh
│   ├── train_model.sh
│   └── uniform_scale_dataset_generation.sh
└── simulator
    ├── dataset_generation
    │   └── generate_dataset.py
    └── temporary_scripts
        ├── elaborate.py
        ├── elaborate_modulate_phase.py
        └── scripts.sh
```

### Data
Download the real recordings from [here](https://doi.org/10.7910/DVN/F0CVOQ) and place them in the folder `forward_model/real_recordings/`.

### Forward Model

<p align="center">
<img src="images/forward_model.png" width="400px"/>  
<br>
Forward Process Layout
</p>

#### Forward Model - Parameter fitting

In order to fit the parameters of the forward model run:
```
python forward_model/forward_model.py -gpu <GPU> -exp_name <NAME OF THE EXPERIMENT> -optimize
```
Where:
- ```<GPU>``` represents which GPU to use
- ```<NAME OF THE EXPERIMENT>``` represents the name to assign to the experiment

#### Forward Model - Synthesis of the dataset for the inverse model
Once the parameters of the forward model have been fitted, you can synthesize a dataset for the inverse model where the pressure is captured by the microphones mounted on a circular array on the aircraft by running the following command:
```
python forward_model/forward_indoor_wrapper.py -gpu <GPU> -exp_name <NAME OF THE EXPERIMENT> -max_order <MAX ORDER> -num_rotors 4 -num_phases_mics <NUMBER OF MICROPHONES> -saved_data_dir_path <DATA PATH> -e_absorption <WALLS ABSORPTION COEFFICIENT> -num_points_per_side <NUMBER OF POINTS PER SIDE> -room_x <WALL 1 LENGTH> -room_y <WALL 2 LENGTH> -number_of_angles <NUMBER OF ANGLES PER POINT>
```
Where:
- ```<GPU>``` represents which GPU to use
- ```<NAME OF THE EXPERIMENT>``` is the name assigned to the experiment during the [parameter fitting](#forward-model---parameter-fitting) phase
- ```<MAX ORDER>``` is the maximum image order to use 
- ```<NUMBER OF MICROPHONES>``` is the number of microphones to use (e.g. 8)
- ```<DATA PATH>``` is the path to save the data
- ```<WALLS ABSORPTION COEFFICIENT>``` is the absorption coefficient of the walls in the room (e.g. 0.5). The absorption coefficient is a value between 0 and 1, where 0 means no absorption and 1 means full absorption. It is defined as $1 - \gamma$, where $\gamma$ is the reflection coefficient.
- ```<NUMBER OF POINTS PER SIDE>``` is the number of points per side in the room (e.g. 63)
- ```<WALL 1 LENGTH>``` is the length of the first wall in meters (assuming a rectangular room)
- ```<WALL 2 LENGTH>``` is the length of the second wall in meters (assuming a rectangular room)
- ```<NUMBER OF ANGLES PER POINT>``` is the number of angles to use per point in the room (e.g. 64)

#### Forward Model - Generate the robustness test datasets
In order to reproduce the datasets used for the robustness tests, first fit the parameters of the forward model (see [parameter fitting](#forward-model---parameter-fitting)) and then run the commands listed in the next sections.

##### Robustness test: uniform scale
To generate the dataset for the robustness test of the uniform scale:
1. Generate the set of RIRs (Room Impulse Responses) for the different scales:
```
scripts/uniform_scale_dataset_generation.sh cpu
```
2. Given the set of RIRs, generate the set of sounds :
```
scripts/uniform_scale_dataset_generation.sh gpu <NUMBER OF GPUS>
```
Where:
- ```<NUMBER OF GPUS>``` is the number of GPUs to use

##### Robustness test: aspect ratio
To generate the dataset for the robustness test of the aspect ratio:
1. Generate the set of RIRs (Room Impulse Responses) for the different aspect ratios:
```
scripts/aspect_ratio_dataset_generation.sh cpu
```
2. Given the set of RIRs, generate the set of sounds :
```
scripts/aspect_ratio_dataset_generation.sh gpu <NUMBER OF GPUS>
```
Where:
- ```<NUMBER OF GPUS>``` is the number of GPUs to use

##### Robustness test: shear
To generate the dataset for the robustness test of the shear:
1. Generate the set of RIRs (Room Impulse Responses) for the different shears:
```
scripts/shear_dataset_generation.sh cpu
```
2. Given the set of RIRs, generate the set of sounds :
```
scripts/shear_dataset_generation.sh gpu <NUMBER OF GPUS>
```
Where:
- ```<NUMBER OF GPUS>``` is the number of GPUs to use

##### Robustness test: acoustic reflection coefficient
To generate the dataset for the robustness test of the acoustic reflection coefficient:
1. Generate the set of RIRs (Room Impulse Responses) for the different reflection coefficients:
```
scripts/acoustic_reflection_coefficient_dataset_generation.sh cpu
```
2. Given the set of RIRs, generate the set of sounds :
```
scripts/acoustic_reflection_coefficient_dataset_generation.sh gpu <NUMBER OF GPUS>
```
Where:
- ```<NUMBER OF GPUS>``` is the number of GPUs to use

### Inverse Model
* Explain how to train inverse model for all experiments



## Acknowledgements 
This research was supported by ERC StG EARS. We are
grateful to Yair Atzmon, Matan Jacoby, Aram Movsisian,
and Alon Gil-Ad for their help with the data acquisition.

## Citation
If you use this code for your research, please cite the following work: 
```
@misc{@article{gabriele2024active,
  title={Active propulsion noise shaping for multi-rotor aircraft localization},
  author={Gabriele, Serussi and Tamir, Shor and Tom, Hirshberg and Chaim, Baskin and Alex, Bronstein},
  journal={arXiv preprint arXiv:2402.17289},
  year={2024}
}
}
```
