import torch
import numpy as np
import itertools
import logging
import pathlib
import random
import shutil
import time
import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt

import sys

from EARS_code.localization.phase_modulation.phase_modulation_pipeline import Localization_Model
from EARS_code.localization.phase_modulation.modulation_dataset import  \
    ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient, \
    ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2d, collate_fn, \
    ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2dOrientation, \
    Localization2dGivenOrientationDataset, collate_fn_orientation
from torch.utils.data import DataLoader, RandomSampler

import torch
import os
from EARS_code.localization.physics import Physics, SAMPLES_PER_ROTATION, PLOT_DT #get_integrated_velocity
from EARS_code.localization.penalty import Penalty #get_integrated_velocity_penalty
#from torchviz import make_dot

from math import sqrt
import h5py

# Initialize logging
from EARS_code.localization.multi_position import master, aggregator, trajectory_factory
import EARS_code.localization.multi_position.dataset as multi_position_dataset

#STABILITY_SCALE_TIME_FACTOR = 0.34


def loss_fn_localization_and_orientation(output, target):
    loss_coordinates = torch.nn.functional.mse_loss(output[..., :-1], target[..., :-1])
    loss_orientation = torch.mean(1 - torch.cos(output[..., -1] - target[..., -1]))
    return loss_coordinates + loss_orientation


def loss_fn_localization(output, target):
    return torch.nn.functional.mse_loss(output[..., :-1], target[..., :-1])


def loss_fn_orientation(output, target):
    return torch.mean(1 - torch.cos(output[..., -1] - target[..., -1]))


def orientation_loss_to_difference_in_degrees(loss_value):
    '''This function takes as input the orientation loss and it returns the difference of angles in degrees (used only for displaying purposes)'''
    return np.rad2deg(np.arccos(1 - np.clip(loss_value, 0, 2)))


def get_vel_acc(x, dt):
    # calculate numerical derivatives of the phase shifts
    #dt = 0.00033291878245085695 / STABILITY_SCALE_TIME_FACTOR  # 0.34 for period time scaling (unit conversion for num stability)

    # prepend is used in order to force circular boundary conditions
    # v = torch.diff(x, n=1, prepend=x[:,-1].unsqueeze(-1), dim=-1)/dt
    v = torch.diff(x, n=1, dim=-1)/dt
    #v = (x[:, 1:] - x[:, :-1]) / dt

    # a = (v[:, 1:] - v[:, :-1]) / dt
    # a = torch.diff(v, n=1, prepend=v[:,-1].unsqueeze(-1), dim=-1)/dt
    a = torch.diff(v, n=1, dim=-1)/dt
    return v, a


def create_data_loaders(args):
    # dataset = AudioLocalizationDataset(args.data_path)
    # dataset = ModulationDatasetFixedInputSound()
    if args.use_2d_given_orientation:
        if args.use_non_convex_room:
            min_coord: torch.Tensor = torch.tensor([0.93, 0.93])
            max_coord: torch.Tensor = torch.tensor([8.07, 6.07])
        elif args.use_asymmetric_non_convex_room:
            min_coord: torch.Tensor = torch.tensor([2.053445041311215, 1.6472153073726603])
            max_coord: torch.Tensor = torch.tensor([7.9, 7.104502353838798])
        else:
            min_coord: torch.Tensor = torch.tensor([0.93, 0.93])
            max_coord: torch.Tensor = torch.tensor([4.07, 4.07])

        if args.use_multi_position:
            compressed_datasets = set(["/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/orig_64_angles/default_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/non_convex_room/non_convex_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/small_dense_room/default_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/shifted_room/default_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/non_convex_room/shifted_non_convex_5.0_5.0_order_1_0.5_d_0.05/",
                                 "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/asymmetric_non_convex_room/non_convex_5.0_5.0_order_1_0.5_d_0.05"
                                 ])
            use_hdf5 = args.use_newton_cluster or args.data_path in compressed_datasets
            if args.trajectory_factory == 'ccw':
                trajectory_fact = trajectory_factory.CCWTrajectoryFactory(use_hdf5=use_hdf5, number_of_points=args.position_number)
            elif args.trajectory_factory == 'all':
                trajectory_fact = trajectory_factory.AllAnglesTrajectoryFactory(use_hdf5=use_hdf5)
            else:
                raise ValueError("Currently the only supported trajectory factory are ccw and all. Please choose one among them")

            if args.test_non_aggregating_trained:
                dataset = Localization2dGivenOrientationDataset(args.data_path, duration=args.duration, filter_angles=args.use_mega_dataset, use_newton_cluster=args.use_newton_cluster,
                                                                min_coord=min_coord, max_coord=max_coord)
            else:
                dataset = multi_position_dataset.MultiPositionDataset(args.data_path, trajectory_fact, duration=args.duration, filter_angles=args.use_mega_dataset, use_newton_cluster=args.use_newton_cluster)
        else:
            dataset = Localization2dGivenOrientationDataset(args.data_path, duration=args.duration, filter_angles=args.use_mega_dataset, use_newton_cluster=args.use_newton_cluster,
                                                            min_coord=min_coord, max_coord=max_coord)
        torch.autograd.set_detect_anomaly(True)
    elif args.use_2d:
        dataset = ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2d(args.data_path, duration=args.duration)
    elif args.use_orientation:
        dataset = ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient2dOrientation(args.data_path, duration=args.duration)
    else:
        dataset = ModulationDatasetFixedInputSoundFixedAbsorptionCoefficient(absorption_coefficient=0.2, duration=args.duration)
    rec_len = dataset.get_rec_len()
    print(f'Using a recording length of {rec_len} (raw samples)')
    if args.local_testing:
        train_dataset = dataset
    else:
        # use 80 train, 10% val, 10% test
        train_split_ratio = 0.8
        val_split_ratio = 0.10
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                [int(train_split_ratio * len(dataset)),
                                                                                int(val_split_ratio * len(dataset)),
                                                                                len(dataset) - int(
                                                                                    train_split_ratio * len(
                                                                                        dataset)) - int(
                                                                                    val_split_ratio * len(dataset))])
        if args.test_non_aggregating_trained:
            train_used_paths = [train_dataset.dataset.single_data_path[i] for i in train_dataset.indices]
            train_trajectory_fact = trajectory_factory.CCWTrajectoryFactory(use_hdf5=use_hdf5, number_of_points=args.position_number)
            train_dataset = multi_position_dataset.MultiPositionDataset(args.data_path, train_trajectory_fact, duration=args.duration, filter_angles=args.use_mega_dataset, use_newton_cluster=args.use_newton_cluster,
                                                                       paths_list=train_used_paths)
            val_used_paths = [val_dataset.dataset.single_data_path[i] for i in val_dataset.indices]
            val_trajectory_fact = trajectory_factory.CCWTrajectoryFactory(use_hdf5=use_hdf5, number_of_points=args.position_number)
            val_dataset = multi_position_dataset.MultiPositionDataset(args.data_path, val_trajectory_fact, duration=args.duration, filter_angles=args.use_mega_dataset, use_newton_cluster=args.use_newton_cluster,
                                                                       paths_list=val_used_paths)
            test_used_paths = [test_dataset.dataset.single_data_path[i] for i in test_dataset.indices]
            test_trajectory_fact = trajectory_factory.CCWTrajectoryFactory(use_hdf5=use_hdf5, number_of_points=args.position_number)
            test_dataset = multi_position_dataset.MultiPositionDataset(args.data_path, test_trajectory_fact, duration=args.duration, filter_angles=args.use_mega_dataset, use_newton_cluster=args.use_newton_cluster,
                                                                       paths_list=test_used_paths)
    if args.use_2d or args.use_2d_given_orientation:
        if args.use_2d_given_orientation:
            if args.use_multi_position:
                fnct = multi_position_dataset.collate_fn_orientation
            else:
                fnct = collate_fn_orientation
        else:
            fnct = collate_fn
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=fnct,
                                    #   pin_memory=True, num_workers=3)
                                      pin_memory=True)
        if args.local_testing:
            val_dataloader = None
            test_dataloader = None
        else:
            if args.test_non_aggregating_trained and args.num_epochs == 1:
                val_dataloader = []
            else:
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=fnct,
                                        pin_memory=True)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=fnct,
                                        pin_memory=True)
    elif args.use_orientation:
        if args.use_subset:
            train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=20_000)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                          pin_memory=True,
                                          sampler=train_sampler)
            val_sampler = RandomSampler(val_dataset, replacement=False, num_samples=20_000)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                        pin_memory=True,
                                        sampler=val_sampler)

            test_sampler = RandomSampler(test_dataset, replacement=False, num_samples=4_000)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                         pin_memory=True,
                                         sampler=test_sampler)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          collate_fn=collate_fn, pin_memory=True)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                         pin_memory=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader, rec_len


def plot_phases(args, phases, best=False):
    fig, axes = plt.subplots(nrows=2, ncols=2)

    for i in range(phases.shape[0]):
        row = i // 2  # Row index of the subplot
        col = i % 2  # Column index of the subplot
        ax = axes[row, col]  # Get the corresponding subplot
        ax.plot(range(phases.shape[-1]), phases[i, :].cpu().detach())  # Plot the data on the subplot
        ax.set_title(f"Rotor {i}")
        ax.set_ylabel("Phase Shift [rad/sec]")
        ax.set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(f"{args.exp_dir}/phase_shift_plot{'_best' if best else ''}.png")
    plt.close(fig=fig)

def plot_system_response(args, model, phases, best=False):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    perturbed_phases, injected_phases = model.get_system_response(phases, args.system_noise_snr_db)

    for i in range(perturbed_phases.shape[0]):
        row = i // 2  # Row index of the subplot
        col = i % 2  # Column index of the subplot
        ax = axes[row, col]  # Get the corresponding subplot
        ax.plot(range(perturbed_phases.shape[-1]), perturbed_phases[i, :].cpu().detach())  # Plot the data on the subplot
        ax.set_title(f"Rotor {i}")
        ax.set_ylabel("Rotor position [rad]")
        ax.set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(f"{args.exp_dir}/system_response_plot{'_best' if best else ''}.png")
    plt.close(fig=fig)
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for i in range(injected_phases.shape[0]):
        row = i // 2  # Row index of the subplot
        col = i % 2  # Column index of the subplot
        ax = axes[row, col]  # Get the corresponding subplot
        ax.plot(range(injected_phases.shape[-1]), injected_phases[i, :].cpu().detach())  # Plot the data on the subplot
        ax.set_title(f"Rotor {i}")
        ax.set_ylabel("Injected phase [rad]")
        ax.set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(f"{args.exp_dir}/injected_phase_plot{'_best' if best else ''}.png")
    plt.close(fig=fig)
    return perturbed_phases


def train_epoch(args, epoch, model, data_loader, loss_fn, optimizer, loader_len):
    duration = args.duration if args.duration is not None else 0.34
    model.train()
    avg_loss = 0.
    if args.use_orientation:
        avg_localization_loss = 0.
        avg_orientation_loss = 0.
    else:
        avg_localization_loss = 0
        avg_orientation_loss = None

    # The interpolation gap is only meaningful if we do not use projection
    # if epoch < 20:
    #    model.phase_model.interp_gap = int(32*args.gap_factor)
    # elif epoch == 20:
    #    model.phase_model.interp_gap = int(16*args.gap_factor)
    # elif epoch == 30:
    #    model.phase_model.interp_gap = int(8*args.gap_factor)
    # elif epoch == 40:
    #    model.phase_model.interp_gap = int(4*args.gap_factor)
    # elif epoch == 50:
    #    model.phase_model.interp_gap = int(2*args.gap_factor)
    # elif epoch == 60:
    #    model.phase_model.interp_gap = int(2*args.gap_factor)

    start_epoch = time.perf_counter()

    for iter, data in data_loader:

        optimizer.zero_grad()

        start = time.time()

        if args.separate_training_iters:
            if (iter % args.separate_iter) < args.separate_iter_reconstruction:
                model.phase_model.amplitudes.requires_grad = True
                if args.separate_training_full:
                    for param in model.backward_model.parameters():
                        param.requires_grad = False
            else:
                if args.separate_training_full:
                    for param in model.backward_model.parameters():
                        param.requires_grad = True

                if args.no_phase_shift_learn:
                    model.phase_model.amplitudes.requires_grad = False
        else:
            if epoch > args.only_recons_epoch or epoch<args.warmup_epoch:
                model.phase_model.amplitudes.requires_grad = False

            elif epoch == args.warmup_epoch: #revert warmup
                model.phase_model.amplitudes.requires_grad = True

            elif args.freeze_iter and (args.joint_epoch < 0 or epoch < args.joint_epoch):
                if iter % args.freeze_iter:
                    model.phase_model.amplitudes.requires_grad = False
                    if args.split_train:
                        for param in model.backward_model.parameters():
                            param.requires_grad = True
                        model.backward_model.train()
                else:
                    if args.no_phase_shift_learn:
                        model.phase_model.amplitudes.requires_grad = True
                    if args.split_train:
                        for param in model.backward_model.parameters():
                            param.requires_grad = False
                        model.backward_model.eval()
        if args.use_2d_given_orientation:
            input, target, orientation = data
            orientation = orientation.to(args.device)
        else:
            input, target = data

        input = input.to(args.device)
        target = target.to(args.device)

        if args.noise:  # add noise to shift initialialization
            noise_factor = 10e-6
            noise_precentage = 0.99

            for j in range(args.num_rotors):
                theta = np.pi / args.n_shots
                for i in range(args.n_shots):
                    Ly = torch.arange(-np.pi, np.pi, 2 * np.pi / input.shape[-1]).float()

                    model.phase_model.x.data[j, i, :] = Ly * np.sin(theta * i)
                    for k in range(input.shape[-1]):
                        num = np.random.rand()
                        if num > noise_precentage:
                            sign = np.random.randint(0, 2)
                            if sign:
                                model.phase_model.x.data[j, i, k] -= np.random.rand() * noise_factor
                            else:
                                model.phase_model.x.data[j, i, k] += np.random.rand() * noise_factor
        # preprocess input

        if args.inject_noise_in_sound:
            if args.desired_snr_in_db is not None:
                if args.use_2d_given_orientation:
                    output = model(input, orientation=orientation, desired_snr_in_db=args.desired_snr_in_db,
                                   system_noise_snr_db=args.system_noise_snr_db)
                else:
                    output = model(input, desired_snr_in_db=args.desired_snr_in_db)
            else:
                # choose random snr from list
                desired_snr_in_db = args.desired_snr_in_db_list[np.random.randint(0, len(args.desired_snr_in_db_list))]
                if args.use_2d_given_orientation:
                    output = model(input, orientation=orientation, desired_snr_in_db=desired_snr_in_db,
                                   system_noise_snr_db=args.system_noise_snr_db)
                else:
                    output = model(input, desired_snr_in_db=desired_snr_in_db)
        else:
            if args.use_2d_given_orientation:
                output = model(input, orientation=orientation,
                               system_noise_snr_db=args.system_noise_snr_db)
            else:
                output = model(input)
        x = model.get_phases()
        # dt=1/model.phase_model.recording_length
        dt = PLOT_DT
        v, a = get_vel_acc(x, dt)
        # acc_loss = torch.sqrt(torch.sum(torch.pow(torch.nn.functional.softshrink(a, args.a_max).abs() + 1e-8, 2)))
        acc_loss = torch.linalg.norm(
            #torch.nn.functional.softshrink(a, args.a_max / (STABILITY_SCALE_TIME_FACTOR ** 2)).abs() + 1e-8, ord=2)
            # torch.nn.functional.softshrink(a, args.a_max / (duration ** 2)).abs() + 1e-8, ord=2)
            torch.nn.functional.softshrink(a, args.a_max).abs() + 1e-8, ord=2)
        # vel_loss = torch.sqrt(torch.sum(torch.pow(torch.nn.functional.softshrink(v, args.v_max).abs() + 1e-8, 2)))
        vel_loss = torch.linalg.norm(
            #torch.nn.functional.softshrink(v, args.v_max / STABILITY_SCALE_TIME_FACTOR).abs() + 1e-8, ord=2)
            # torch.nn.functional.softshrink(v, args.v_max / duration).abs() + 1e-8, ord=2)
            torch.nn.functional.softshrink(v, args.v_max).abs() + 1e-8, ord=2)

        if args.window_type == 'ones':
            vel_integrated = Physics(recording_length=model.phase_model.recording_length).\
                get_integrated_velocity(x, args.integrated_velocity_window_size, window_type='')
            vel_integrated_loss = torch.linalg.norm(
                torch.nn.functional.softshrink(vel_integrated, args.v_integrated_max).abs() + 1e-8, ord=2)
        elif args.window_type == 'gaussian':
            '''
            vel_integrated_loss = get_integrated_velocity_penalty(model.phase_model.amplitudes.weight, fwhm_list=args.fwhm,
                                                                  basis_size=args.fourier_basis_size, cutoff=args.fourier_cutoff_freq,
                                                                  rec_len=model.phase_model.recording_length,
                                                                  num_rotors=args.num_rotors)
            '''
            vel_integrated_loss = Penalty(physics=Physics(recording_length=model.phase_model.recording_length)).\
                get_integrated_velocity_penalty(model.phase_model.amplitudes, fwhm_list=args.fwhm,
                                                                  basis_size=args.fourier_basis_size,
                                                                  cutoff=args.fourier_cutoff_freq,
                                                                  rec_len=model.phase_model.recording_length,
                                                                  num_rotors=args.num_rotors)
        eps = 10e-3
        if not args.use_orientation:
            # target loss
            loc_loss = loss_fn(output.to(torch.float64), target.to(torch.float64))
            avg_localization_loss += (loc_loss.item() - avg_localization_loss) / (iter+1)
            #image_name = "loc_loss_sound_transformer"
            #make_dot(loc_loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render(image_name, format="png")
            #exit()
            # weigh kinematic los in overall loss
            loss = args.rec_weight * loc_loss + args.vel_weight * vel_loss + args.acc_weight * acc_loss + \
                   args.v_integrated_weight * vel_integrated_loss + args.amp_penalty_coeff/(eps + torch.linalg.norm(model.phase_model.amplitudes[0],2))
        else:
            loc_loss = loss_fn_localization(output.to(torch.float64), target.to(torch.float64))
            avg_localization_loss += (loc_loss.item() - avg_localization_loss) / (iter + 1)
            orientation_loss = loss_fn_orientation(output.to(torch.float64), target.to(torch.float64))
            avg_orientation_loss += (orientation_loss.item() - avg_orientation_loss) / (iter + 1)
            # weigh kinematic los in overall loss
            loss = args.rec_weight * loc_loss + args.orientation_weight * orientation_loss + \
                   args.vel_weight * vel_loss + args.acc_weight * acc_loss
        try:
            loss.backward()
        except:
            print(f"output {output} target {target}")
            print(f"loc loss {loc_loss} vel loss {vel_loss} acc loss {acc_loss} vel integrated loss {vel_integrated_loss}")
            print(f"norm loss {torch.linalg.norm(model.phase_model.amplitudes[0],2)}")
            print(f"amplitudes {model.phase_model.amplitudes} (shape {model.phase_model.amplitudes.shape})")
            raise ValueError()
        optimizer.step()

        # running average
        # avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        # arithmetic average
        avg_loss += (loss.item() - avg_loss) / (iter + 1)

        if iter % args.report_interval == 0:
            if args.use_orientation:
                logging.info(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                    f'Iter = [{iter:4d}/{loader_len:4d}] '
                    f' Avg Loss {avg_loss:.4g} LocalizationLoss = {sqrt(avg_localization_loss):.4g} Orientation Diff = {orientation_loss_to_difference_in_degrees(avg_orientation_loss):.4g} (deg)'
                )
            else:
                logging.info(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                    f'Iter = [{iter:4d}/{loader_len:4d}] '
                    f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                    f'LocLoss = {sqrt(loc_loss)} Avg LocLoss = {sqrt(avg_localization_loss)} '
                    f'vel loss {vel_loss:.4g}, acc loss {acc_loss:.4g} vel_integrated loss {vel_integrated_loss:.4g}'
                )
            plot_phases(args, x)
            if args.simulate_system:
                perturbed_phases = plot_system_response(args, model, x)
                torch.save(perturbed_phases, args.exp_dir + '/current_perturbed_phases.pt')
            torch.save(model.phase_model.amplitudes, args.exp_dir + '/current_phase_amplitude.pt')
        if iter == loader_len - 1:
            break

    return avg_loss, time.perf_counter() - start_epoch, loc_loss, vel_loss, acc_loss, avg_localization_loss, avg_orientation_loss


def evaluate(args, epoch, model, data_loader, loss_fn, dl_len, train_loss=None, train_rec_loss=None,
             train_vel_loss=None,
             train_acc_loss=None, inject_noise_in_sound:bool=False):
    model.eval()
    losses = []
    if args.use_orientation:
        localization_losses = []
        orientation_losses = []
    start = time.perf_counter()
    with torch.no_grad():
        if epoch != 0:
            for iter, data in data_loader:

                # if args.use_2d:
                #     input = torch.tensor(data)
                #     target = torch.tensor(data)
                # else:
                #     input, target = data
                #     input = input.to(args.device)
                #     target = target.to(args.device)
                if args.use_2d_given_orientation:
                    input, target, orientation = data
                    orientation = orientation.to(args.device)
                else:
                    input, target = data
                input = input.to(args.device)
                target = target.to(args.device)

                if inject_noise_in_sound:
                    if args.desired_snr_in_db is not None:
                        if args.use_2d_given_orientation:
                            output = model(input, orientation=orientation, desired_snr_in_db=args.desired_snr_in_db,
                                           system_noise_snr_db=args.system_noise_snr_db)
                        else:
                            output = model(input, desired_snr_in_db=args.desired_snr_in_db)
                    else:
                        # choose random snr from list
                        desired_snr_in_db = args.desired_snr_in_db_list[
                            np.random.randint(0, len(args.desired_snr_in_db_list))]
                        if args.use_2d_given_orientation:
                            output = model(input, orientation=orientation, desired_snr_in_db=desired_snr_in_db,
                                           system_noise_snr_db=args.system_noise_snr_db)
                        else:
                            output = model(input, desired_snr_in_db=args.desired_snr_in_db)
                else:
                    if args.use_2d_given_orientation:
                        output = model(input, orientation=orientation,
                                       system_noise_snr_db=args.system_noise_snr_db)
                    else:
                        output = model(input)

                if args.use_orientation:
                    loc_loss = loss_fn_localization(output.to(torch.float64), target.to(torch.float64))
                    localization_losses.append(loc_loss)
                    orientation_loss = loss_fn_orientation(output.to(torch.float64), target.to(torch.float64))
                    orientation_losses.append(orientation_loss)
                    loss = args.rec_weight * loc_loss + args.orientation_weight * orientation_loss
                else:
                    loss = loss_fn(output, target)
                losses.append(loss.item())

                if iter == dl_len - 1:
                    break

            x = model.get_phases()
            v, a = get_vel_acc(x, dt=1/model.phase_model.recording_length)

            # #acc_loss = torch.sqrt(torch.sum(torch.pow(torch.nn.functional.softshrink(a, args.a_max), 2)))
            # acc_loss = torch.linalg.norm(torch.nn.functional.softshrink(a, args.a_max).abs()+1e-8, ord=2)
            # # vel_loss = torch.sqrt(torch.sum(torch.pow(torch.nn.functional.softshrink(v, args.v_max), 2)))
            # vel_loss = torch.linalg.norm(torch.nn.functional.softshrink(v, args.v_max).abs()+1e-8, ord=2)
            # rec_loss = np.mean(losses)

            if args.window_type == 'ones':
                vel_integrated = Physics(recording_length=model.phase_model.recording_length).get_integrated_velocity(x, args.integrated_velocity_window_size, window_type='')
                vel_integrated_loss = torch.linalg.norm(
                    torch.nn.functional.softshrink(vel_integrated, args.v_integrated_max).abs() + 1e-8, ord=2)
            elif args.window_type == 'gaussian':
                # vel_integrated_loss = get_integrated_velocity_penalty(model.phase_model.amplitudes.weight, fwhm_list=args.fwhm)
                '''
                vel_integrated_loss = get_integrated_velocity_penalty(model.phase_model.amplitudes.weight, fwhm_list=args.fwhm,
                                                                  basis_size=args.fourier_basis_size, cutoff=args.fourier_cutoff_freq,
                                                                  rec_len=model.phase_model.recording_length,
                                                                  num_rotors=args.num_rotors)
                '''
                # vel_integrated_loss = get_integrated_velocity_penalty(model.phase_model.amplitudes, fwhm_list=args.fwhm,
                #                                                       basis_size=args.fourier_basis_size,
                #                                                       cutoff=args.fourier_cutoff_freq,
                #                                                       rec_len=model.phase_model.recording_length,
                #                                                       num_rotors=args.num_rotors)
                vel_integrated_loss = Penalty(physics=Physics(recording_length=model.phase_model.recording_length)).\
                get_integrated_velocity_penalty(model.phase_model.amplitudes, fwhm_list=args.fwhm,
                                                                  basis_size=args.fourier_basis_size,
                                                                  cutoff=args.fourier_cutoff_freq,
                                                                  rec_len=model.phase_model.recording_length,
                                                                  num_rotors=args.num_rotors)

        x = model.get_phases()
        #dt=1/model.phase_model.recording_length
        dt = PLOT_DT
        v, a = get_vel_acc(x, dt)

    if args.use_orientation:
        if epoch == 0:
            return None, time.perf_counter() - start, None, None, None
        else:
            return np.mean(losses), time.perf_counter() - start, np.mean(localization_losses), np.mean(
                orientation_losses),np.std(losses)
    else:
        if epoch == 0:
            return None, time.perf_counter() - start, None, None
        else:
            return np.mean(losses), time.perf_counter() - start, np.std(losses), None


class DataParallelModule(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, name=None,
               use_newton_cluster=False):
    if use_newton_cluster:
        # Save main model checkpoint in HDF5 format
        with h5py.File((exp_dir + '/model.hdf5') if name is None else name, 'w') as hdf5_file:
            hdf5_file.create_dataset('epoch', data=epoch)
            hdf5_file.create_dataset('args', data=args)
            hdf5_file.create_dataset('best_dev_loss', data=best_dev_loss)
            hdf5_file.create_dataset('exp_dir', data=exp_dir)

            # Save model state dict
            model_state_dict = model.state_dict()
            for key in model_state_dict:
                hdf5_file.create_dataset(f'model/{key}', data=model_state_dict[key])

            # Save optimizer state dict
            optimizer_state_dict = optimizer.state_dict()
            for key in optimizer_state_dict:
                hdf5_file.create_dataset(f'optimizer/{key}', data=optimizer_state_dict[key])

        # Save additional files
        torch.save(model.phase_model.amplitudes, exp_dir + 'phase_amplitude.pt')

        if is_new_best:
            plot_phases(args, model.get_phases(), best=True)
            if args.simulate_system:
                perturbed_phases = plot_system_response(args, model, model.get_phases(), best=True)
                # Save best perturbed phases in HDF5 format
                with h5py.File(exp_dir + 'best_perturbed_phases.hdf5', 'w') as hdf5_file:
                    hdf5_file.create_dataset('best_perturbed_phases', data=perturbed_phases)

            # Save best phase modulation in HDF5 format
            with h5py.File(exp_dir + 'best_phase_modulation.hdf5', 'w') as hdf5_file:
                hdf5_file.create_dataset('best_phase_modulation', data=model.get_phases())
            
            

            # Copy model checkpoint to best_model.pt
            shutil.copyfile(exp_dir + '/model.hdf5', exp_dir + '/best_model.hdf5')

            # Save best phase amplitude
            torch.save(model.phase_model.amplitudes, exp_dir + 'best_phase_amplitude.pt')
    else:
        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dev_loss': best_dev_loss,
                'exp_dir': exp_dir
            },
            f=(exp_dir + '/model.pt') if name is None else name
        )
        torch.save(model.phase_model.amplitudes, exp_dir + 'phase_amplitude.pt')
        if is_new_best:
            plot_phases(args, model.get_phases(), best=True)
            if args.simulate_system:
                perturbed_phases = plot_system_response(args, model, model.get_phases(), best=True)
                torch.save(perturbed_phases, exp_dir + 'best_perturbed_phases.pt')
            torch.save(model.get_phases(), exp_dir + 'best_phase_modulation.pt')
            shutil.copyfile(exp_dir + '/model.pt', exp_dir + '/best_model.pt')
            shutil.copyfile(exp_dir + 'phase_amplitude.pt', exp_dir + 'best_phase_amplitude.pt')
            #torch.save(model.phase_model.amplitudes, exp_dir + 'best_phase_amplitude.pt')


def build_model(args, rec_len):
    if args.use_2d or args.use_2d_given_orientation:
        num_coordinates = 2
    elif args.use_orientation:
        num_coordinates = 3
    else:
        num_coordinates = 1

    model = Localization_Model(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_microphones=args.num_mics,
        dropout=args.drop_prob,
        decimation_rate=args.decimation_rate,
        max_vel=args.v_max,
        max_acc=args.a_max,
        res=args.resolution,
        learn_shift=args.no_phase_shift_learn,
        initialization=args.initialization,
        n_rotors=args.num_rotors,
        interp_gap=args.interp_gap,
        rec_len=rec_len,
        projection_iters=args.proj_iters,
        project=args.project,
        device=args.device,
        no_fourier=args.no_fourier,
        inject_noise=args.inject_noise,
        standardize=args.standardize,
        num_coordinates=num_coordinates,
        use_cnn=args.use_cnn,
        use_wide_cnn=args.use_wide_cnn,
        use_fourier=args.use_fourier,
        cutoff=args.fourier_cutoff_freq,
        basis_size=args.fourier_basis_size,
        use_cosine=args.use_cosine_basis,
        high_resolution_stft=args.high_resolution_stft,
        sampling_interpolation_method = args.sampling_interpolation_method,
        phase_modulation_snr_db_list = args.phase_modulation_snr_db_list,
        init_size = args.init_size,
        use_aggregate=args.use_aggregate,
        use_sound_transformer=args.use_sound_transformer,
        use_orientation_aggregate=args.use_2d_given_orientation,
        use_orientation_transformer=args.use_orientation_transformer,
        simulate_system=args.simulate_system,
        use_orientation_aggregate_mlp=args.use_orientation_aggregate_mlp,
        use_orientation_aggregate_attention=args.use_orientation_aggregate_attention,
        #use_fast_system=args.use_fast_system,
        simulate_system_method="fourier" if not hasattr(args, "simulate_system_method") else args.simulate_system_method,
    ).to(args.device).double()
    model.phase_model.interp_gap = int(32 * args.gap_factor)

    if hasattr(args, "use_multi_position") and args.use_multi_position:
        if args.aggregator == 'deep':
            model = master.DeepRobustMultiPositionModel(single_position_model=model,
                    dim_feedforward=2048 if not hasattr(args, "aggregator_dim_feedforward") else args.aggregator_dim_feedforward,
                    num_heads=8 if not hasattr(args, "aggregator_num_heads") else args.aggregator_num_heads,
                    dropout=0.2 if not hasattr(args, "aggregator_dropout") else args.aggregator_dropout,
                    num_layers=3 if not hasattr(args, "aggregator_num_layers") else args.aggregator_num_layers).to(args.device).double()
        else:
            if args.aggregator == 'geometric_median':
                aggregator_model = aggregator.GeometricMedianAggregator()
            elif args.aggregator == 'mean':
                aggregator_model = aggregator.AverageAggregator()
            elif args.aggregator == 'mlp':
                aggregator_model = aggregator.MLPAggregator(args.position_number)
            elif args.aggregator == 'transformer':
                aggregator_model = aggregator.TransformerAggregator()
            else:
                raise ValueError("Please choose a valid aggregator by setting --aggregator to an option among 'mean', 'mlp' and 'transformer'")
            model = master.ParallelMultiPositionModel(single_position_model=model, aggregator=aggregator_model)
            model = model.to(args.device).double()
        print(f"[debug] Built a multiposition model: {model}")

    return model


def load_model(checkpoint_file, rec_len,args):
    
    if True or not args.use_newton_cluster:
        checkpoint = torch.load(checkpoint_file)
        # #args = checkpoint['args']
        # model = build_model(args, rec_len)
        # if args.data_parallel:
        #     # model = torch.nn.DataParallel(model)
        #     model = DataParallelModule(model)
        # model.load_state_dict(checkpoint['model'])
        # model.phase_model.x = model.phase_model.basis2curve()
        # t = torch.arange(0, model.phase_model.x.shape[1], device=model.phase_model.x.device).float()
        # t1 = t[::32]
        # x_short = model.phase_model.x[:, ::32]
        # for rotor in range(model.phase_model.x.shape[0]):
        #     model.phase_model.x.data[rotor, :] = model.phase_model.interp(t1, x_short[rotor, :], t)

        # if not (use_multi_position and aggregator==deep), then I need to follow classical
        #   else 
        #      if I'm loading just the single position model, then I need to wrap it with the aggregator similarly to test_non_aggregating_trained
        #     else, I need to load everything combined from the aggregator checkpoint

        old_arguments = checkpoint['args']
        model = build_model(old_arguments, rec_len)
        if old_arguments.data_parallel:
            model = DataParallelModule(model)

        if args.test_non_aggregating_trained:
            if args.aggregator == 'mean':
                aggregator_model = aggregator.AverageAggregator()
            elif args.aggregator == 'mlp':
                aggregator_model = aggregator.MLPAggregator(args.position_number)
            elif args.aggregator == 'transformer':
                aggregator_model = aggregator.TransformerAggregator()
            else:
                raise ValueError("Please choose a valid aggregator by setting --aggregator to an option among 'mean', 'mlp' and 'transformer'")
            model = master.ParallelMultiPositionModel(single_position_model=model, aggregator=aggregator_model)
            model = model.to(args.device).double()
            model.single_position_model.load_state_dict(checkpoint['model'])
            for param in itertools.chain(model.single_position_model.backward_model.parameters(),
                                         model.single_position_model.phase_model.parameters()): #model.single_position_model.parameters():
                param.requires_grad = False
            model.single_position_model.eval()
        else:
            if not old_arguments.use_multi_position and args.use_multi_position:
                # we are finetuning a pretrained model
                if args.aggregator == 'deep':
                    model = master.DeepRobustMultiPositionModel(single_position_model=model,
                            dim_feedforward=args.aggregator_dim_feedforward,
                            num_heads=args.aggregator_num_heads,
                            dropout=args.aggregator_dropout,
                            num_layers=args.aggregator_num_layers).to(args.device).double()
                else:
                    if args.aggregator == 'geometric_median':
                        aggregator_model = aggregator.GeometricMedianAggregator()
                    elif args.aggregator == 'mean':
                        aggregator_model = aggregator.AverageAggregator()
                    elif args.aggregator == 'mlp':
                        aggregator_model = aggregator.MLPAggregator(args.position_number)
                    elif args.aggregator == 'transformer':
                        aggregator_model = aggregator.TransformerAggregator()
                    else:
                        raise ValueError("Please choose a valid aggregator by setting --aggregator to an option among 'mean', 'mlp' and 'transformer'")
                    model = master.ParallelMultiPositionModel(single_position_model=model, aggregator=aggregator_model)
                    model = model.to(args.device).double()
                #print(f"[debug] Built a multiposition model: {model}")

            model.load_state_dict(checkpoint['model'])


        optimizer = build_optim(args, model)
        if args.test_non_aggregating_trained:
            pass
        else:
            #optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_load_state_dict(optimizer, checkpoint['optimizer'], model, args)
        return checkpoint, model, optimizer
    else:
        checkpoint = {}
        with h5py.File(checkpoint_file, 'r') as hdf5_file:
            epoch = hdf5_file['epoch'][()]
            checkpoint['epoch'] = epoch
            old_arguments = hdf5_file['args'][()]
            # best_dev_loss = hdf5_file['best_dev_loss'][()]
            # exp_dir = hdf5_file['exp_dir'][()]

            # Load model state dict
            model_state_dict = {}
            for key in hdf5_file['model']:
                model_state_dict[key] = torch.as_tensor(hdf5_file[f'model/{key}'][()])

            # Load optimizer state dict
            optimizer_state_dict = {}
            for key in hdf5_file['optimizer']:
                optimizer_state_dict[key] = torch.tensor(hdf5_file[f'optimizer/{key}'][()])

        model = build_model(old_arguments, rec_len)
        if old_arguments.data_parallel:
            model = DataParallelModule(model)
        model.load_state_dict(model_state_dict)

        optimizer = build_optim(args, model)
        optimizer.load_state_dict(optimizer_state_dict)

        return checkpoint, model, optimizer

def optimizer_load_state_dict(optimizer, state_dict, model, args):
    if len(state_dict['param_groups']) == 2:
        if args.aggregator == 'deep':
            # delete the last 2 parameters which are the ones of the linear layer
            param_indices = list(state_dict['state'].keys())[-2:]
            for param_index in param_indices:
                state_dict['state'].pop(param_index)
                state_dict['param_groups'][1]['params'].remove(param_index)
            
            optimizer.load_state_dict(state_dict)
            param_groups_to_add = [{'params': model.robust_aggregator.parameters(), 'lr': args.lr_aggregator},
                            {'params': model.linear.parameters(), 'lr': args.lr_aggregator}]
            for param_group_to_add in param_groups_to_add:
                optimizer.add_param_group(param_group_to_add)
        else:
            optimizer.load_state_dict(state_dict)

    elif len(state_dict['param_groups']) == 4:
        param_groups_to_add = [{'params': model.robust_aggregator.parameters(), 'lr': args.lr_aggregator},
                        {'params': model.linear.parameters(), 'lr': args.lr_aggregator}]
        for param_group_to_add in param_groups_to_add:
            optimizer.add_param_group(param_group_to_add)
        optimizer.load_state_dict(state_dict)
    else:
        raise ValueError(f"The optimizer checkpoint has an unknown amount of param groups (Got {len(state_dict['param_groups'])} instead of 2 or 4)")
def build_optim(args, model):
    optimizer_arguments = [{'params': model.phase_model.parameters(), 'lr': args.sub_lr},
                            {'params': model.backward_model.parameters()}]

    if args.use_sgd:
        optimizer = torch.optim.SGD(optimizer_arguments, args.lr)
    else:
        if args.test_non_aggregating_trained:
            optimizer = torch.optim.AdamW([{'params': model.aggregator.parameters(), 'lr': args.lr},
            ])
        else:
            optimizer = torch.optim.Adam(optimizer_arguments, args.lr)
    return optimizer


def train(args):
    if args.aggregator == 'deep':
        args.exp_dir = f'summary/deep_aggregator_{args.test_name}'
    else:
        args.exp_dir = f'summary/{args.test_name}'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pathlib.Path(args.exp_dir).mkdir(parents=True, exist_ok=True)

    args_path = 'summary/test/args.txt' if args.test_non_aggregating_trained else args.exp_dir + '/args.txt'
    with open(args_path, "w") as text_file:
        print(vars(args), file=text_file)

    train_loader, dev_loader, test_loader, rec_len = create_data_loaders(args)

    if args.resume:  # load trained model (for evaluation or to keep training)
        checkpoint, model, optimizer = load_model(args.checkpoint, rec_len,args)
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        model = build_model(args, rec_len)
        if args.data_parallel:
            # model = torch.nn.DataParallel(model)
            model = DataParallelModule(model)
        optimizer = build_optim(args, model)
        start_epoch = 0
    logging.info(args)
    if args.use_l1:
        loss_fn = torch.nn.L1Loss().double()
    elif args.use_orientation:
        loss_fn = None  # loss_fn_orientation
    else:
        loss_fn = torch.nn.MSELoss().double()

    enum_train = itertools.cycle(enumerate(train_loader))
    if args.local_testing:
        enum_val = None
    else:
        enum_val = itertools.cycle(enumerate(dev_loader))
        enum_test = itertools.cycle(enumerate(test_loader))

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    best_dev_loss = float('inf')

    args.freeze_iter = 0 if (args.separate_training or args.separate_training_iters) else args.freeze_iter

    for epoch in range(start_epoch, args.num_epochs):

        start = time.time()

        if epoch and not epoch % args.lr_step_size:
            optimizer.param_groups[1]['lr'] *= args.lr_gamma

        if epoch and not epoch % args.sub_lr_time:
            optimizer.param_groups[0]['lr'] = max(args.sub_lr_stepsize * optimizer.param_groups[0]['lr'], 1e-5)

        if args.recons_reset_epoch_freq > 0 and epoch and (
                args.ignore_resets < 0 or epoch <= args.ignore_resets) and not epoch % args.recons_reset_epoch_freq:
            model.recons_reset()

        if not (args.separate_training or args.separate_training_iters):
            if epoch > args.only_recons_epoch:
                model.phase_model.amplitudes.requires_grad = False


            elif (args.joint_epoch < 0 and args.freeze_epoch > 1) or epoch < args.joint_epoch:
                if epoch % args.freeze_epoch:
                    model.phase_model.amplitudes.requires_grad = False
                    if args.split_train:
                        for param in model.backward_model.parameters():
                            param.requires_grad = True
                        model.backward_model.train()

                else:
                    if args.no_phase_shift_learn:
                        model.phase_model.amplitudes.requires_grad = True
                    if args.split_train:
                        for param in model.backward_model.parameters():
                            param.requires_grad = False
                        model.backward_model.eval()
            if epoch == args.joint_epoch:
                if args.no_phase_shift_learn:
                    model.phase_model.amplitudes.requires_grad = True
                for param in model.backward_model.parameters():
                    param.requires_grad = True
                model.backward_model.train()
        else:
            if args.separate_training:
                if (epoch % args.separate_epoch) < args.separate_epoch_reconstruction:
                    model.phase_model.amplitudes.requires_grad = True
                    if args.separate_training_full:
                        for param in model.backward_model.parameters():
                            param.requires_grad = False

                else:
                    if args.no_phase_shift_learn:
                        model.phase_model.amplitudes.requires_grad = False
                        if args.separate_training_full:
                            for param in model.backward_model.parameters():
                                param.requires_grad = True


        if args.use_orientation:
            train_loss, train_time, train_rec_loss, train_vel_loss, train_acc_loss, avg_train_localization_loss, avg_train_orientation_loss = train_epoch(
                args, epoch,
                model,
                enum_train,
                loss_fn,
                optimizer,
                len(train_loader))
        else:
            train_loss, train_time, train_rec_loss, train_vel_loss, train_acc_loss, _, _ = train_epoch(
                args, epoch,
                model,
                enum_train,
                loss_fn,
                optimizer,
                len(train_loader))

        if args.use_orientation:
            if args.inject_noise_in_sound:
                noise_dev_loss, noise_dev_time, noise_dev_loc_loss, noise_dev_orientation_loss,noise_dev_std = evaluate(args, epoch + 1, model, enum_val, loss_fn,
                                                                              len(dev_loader),
                                                                              train_loss,
                                                                              train_rec_loss, train_vel_loss,
                                                                              train_acc_loss, inject_noise_in_sound=True)
                dev_loss, dev_time, dev_loc_loss, dev_orientation_loss,dev_std = evaluate(args, epoch + 1, model, enum_val, loss_fn,
                                                                              len(dev_loader),
                                                                              train_loss,
                                                                              train_rec_loss, train_vel_loss,
                                                                              train_acc_loss, inject_noise_in_sound=False)
            else:
                dev_loss, dev_time, dev_loc_loss, dev_orientation_loss,dev_std = evaluate(args, epoch + 1, model, enum_val, loss_fn,
                                                                                len(dev_loader),
                                                                                train_loss,
                                                                                train_rec_loss, train_vel_loss,
                                                                                train_acc_loss, inject_noise_in_sound=False)
        else:
            if args.local_testing:
                print("Finished testing. Exiting")
                exit()
            else:
                if args.inject_noise_in_sound:
                    noise_dev_loss, noise_dev_time, noise_dev_std, _ = evaluate(args, epoch + 1, model, enum_val, loss_fn,
                                                        len(dev_loader),
                                                        train_loss,
                                                        train_rec_loss, train_vel_loss, train_acc_loss, inject_noise_in_sound=True)
                    
                    dev_loss, dev_time, dev_std, _ = evaluate(args, epoch + 1, model, enum_val, loss_fn,
                                                        len(dev_loader),
                                                        train_loss,
                                                        train_rec_loss, train_vel_loss, train_acc_loss, inject_noise_in_sound=False)
                else:
                    dev_loss, dev_time, dev_std, _ = evaluate(args, epoch + 1, model, enum_val, loss_fn,
                                                        len(dev_loader),
                                                        train_loss,
                                                        train_rec_loss, train_vel_loss, train_acc_loss, inject_noise_in_sound=False)

        if dev_loss < best_dev_loss:
            is_new_best = True
            best_dev_loss = dev_loss
            best_epoch = epoch + 1
        else:
            is_new_best = False
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        if args.use_orientation:
            logging.info(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} Train LocalizationLoss = {sqrt(avg_train_localization_loss):.4g} Train Orientation Diff = {orientation_loss_to_difference_in_degrees(avg_train_orientation_loss)} (deg) TrainTime = {train_time:.4f} s '
                f'DevLoss = {dev_loss:.4g} Dev LocalizationLoss = {sqrt(dev_loc_loss):.4g} Dev Orientation Diff = {orientation_loss_to_difference_in_degrees(dev_orientation_loss)} (deg) DevTime = {dev_time:.4f} s '
                f'DevStd = {dev_std:.4g}'
            )
            if args.inject_noise_in_sound:
                logging.info(
                    f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} Train LocalizationLoss = {sqrt(avg_train_localization_loss):.4g} Train Orientation Diff = {orientation_loss_to_difference_in_degrees(avg_train_orientation_loss)} (deg) TrainTime = {train_time:.4f} s '
                    f'NoiseDevLoss = {noise_dev_loss:.4g} NoiseDev LocalizationLoss = {sqrt(noise_dev_loc_loss):.4g} NoiseDev Orientation Diff = {orientation_loss_to_difference_in_degrees(noise_dev_orientation_loss)} (deg) DevTime = {noise_dev_time:.4f}s'
                    f'DevStd = {noise_dev_std:.4g}'
                )
        else:
            logging.info(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'DevLoss = {sqrt(dev_loss):.4g} rms TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f} s '
                f'DevStd = {dev_std:.4g}'
            )
            if args.inject_noise_in_sound:
                logging.info(
                    f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                    f'NoiseDevLoss = {sqrt(noise_dev_loss):.4g} rms TrainTime = {train_time:.4f}s NoiseDevTime = {noise_dev_time:.4f} s '
                    f'NoiseDevStd = {noise_dev_std:.4g}'
                )

        end = time.time() - start
        print(f'epoch time: {end}')
    

    # Test on test set

    # Convert current model's parameters to the best model's parameters
    if not (args.test_non_aggregating_trained and args.num_epochs==1):
        # best_parameters = torch.load(args.exp_dir+ '/best_model.pt')
        best_parameters = torch.load(os.path.join(args.exp_dir, 'best_model.pt'))
        model.load_state_dict(best_parameters['model'])
    model.eval()
    model = model.double()
    if args.use_orientation:
        test_loss, test_time, test_loc_loss, test_orientation_loss,test_std = evaluate(args, epoch + 1, model, enum_test, loss_fn,
                                                                            len(test_loader),
                                                                            train_loss,
                                                                            train_rec_loss, train_vel_loss,
                                                                            train_acc_loss, inject_noise_in_sound=False)
        logging.info(
                f'TestLoss = {test_loss:.4g} Test LocalizationLoss = {sqrt(test_loc_loss):.4g} Test Orientation Diff = {orientation_loss_to_difference_in_degrees(test_orientation_loss)} (deg) TestTime = {test_time:.4f} s '
                f'TestStd = {test_std:.4g}'
            )
        if args.inject_noise_in_sound:
            noise_test_loss, noise_test_time, noise_test_loc_loss, noise_test_orientation_loss, noise_test_std = evaluate(args, epoch + 1, model, enum_test, loss_fn,
                                                                            len(test_loader),
                                                                            train_loss,
                                                                            train_rec_loss, train_vel_loss,
                                                                            train_acc_loss, inject_noise_in_sound=True)
            logging.info(
                    f'NoiseTestLoss = {noise_test_loss:.4g} NoiseTest LocalizationLoss = {sqrt(noise_test_loc_loss):.4g} NoiseTest Orientation Diff = {orientation_loss_to_difference_in_degrees(noise_test_orientation_loss)} (deg) NoiseTestTime = {noise_test_time:.4f} s '
                    f'NoiseTestStd = {noise_test_std:.4g}'
                )
    else:
        if args.test_non_aggregating_trained and args.num_epochs == 1:
            epoch:int = 1
            train_loss:float = 0.0
            train_time:float = 0.0
            train_loss = None
            train_rec_loss = None
            train_vel_loss = None
            train_acc_loss = None


        test_loss, test_time, test_std, _ = evaluate(args, epoch, model, enum_test, loss_fn,
                                            len(test_loader),
                                            train_loss,
                                                train_rec_loss, train_vel_loss, train_acc_loss, inject_noise_in_sound=False)
        logging.info(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'TestLoss = {sqrt(test_loss):.4g} rms TrainTime = {train_time:.4f}s TestTime = {test_time:.4f} s '
                f'TestStd = {test_std:.4g}'
            )
        if args.inject_noise_in_sound:
            noise_test_loss, noise_test_time, noise_test_std, _ = evaluate(args, epoch + 1, model, enum_test, loss_fn,
                                            len(test_loader),
                                            train_loss,
                                                train_rec_loss, train_vel_loss, train_acc_loss, inject_noise_in_sound=True)
            logging.info(
                    f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                    f'NoiseTestLoss = {sqrt(noise_test_loss):.4g} rms TrainTime = {train_time:.4f}s NoiseTestTime = {noise_test_time:.4f} s '
                    f'NoiseTestStd = {noise_test_std:.4g}'
                )
    


def run():
    args = create_arg_parser().parse_args()
    check_args(args)
    # if args.local_testing:
    #     log_file = "train.log"
    # else:
        # root_log_file = os.path.join("/mnt/walkure_public/tamirs/train_log/", f"{datetime.now()}")
        # if not os.path.exists(root_log_file):
        #     os.makedirs(root_log_file)
        # args.exp_dir = os.path.join(root_log_file, 'summary', f'{args.test_name}')
        # log_specific = "train.log"
        # log_file = os.path.join(root_log_file, log_specific)
    
    log_file = "train.log"
    logging.basicConfig(level=logging.INFO, filename=log_file)
    logger = logging.getLogger(__name__)
    logging.info(f"PID: {os.getpid()}")
    if args.use_2d_given_orientation and args.data_path == '/home/tamir.shor/EARS/data/merged.npy':
        if args.use_asymmetric_non_convex_room:
            args.data_path = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/asymmetric_non_convex_room/non_convex_5.0_5.0_order_1_0.5_d_0.05"
        elif args.finetune:
            args.data_path = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/shifted_room/default_5.0_5.0_order_1_0.5_d_0.05/"
        elif args.use_non_convex_room:
            args.data_path = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/non_convex_room/non_convex_5.0_5.0_order_1_0.5_d_0.05/"
        elif args.use_orig_64_angles:
            args.data_path = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/orig_64_angles/default_5.0_5.0_order_1_0.5_d_0.05/"
        elif args.use_newton_cluster:
            args.data_path = "/home/gabrieles/EARS/data/pressure_field_orientation_dataset/32_angles/default_5.0_5.0_order_1_0.5_d_0.05/"
        elif args.use_big_dataset:
            args.data_path = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/big_dataset/default_5.0_5.0_order_1_0.5_d_0.05/"
        elif args.use_mega_dataset:
            args.data_path = "/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/mega_dataset/default_5.0_5.0_order_1_0.5_d_0.05/"
        else:
            args.data_path = '/mnt/walkure_public/tamirs/pressure_field_orientation_dataset/default_5.0_5.0_order_1_0.5_d_0.05'
    # if set 2d and data_path is the default one, change it to the 2d one
    elif args.use_2d and args.data_path == '/home/tamir.shor/EARS/data/merged.npy':
        # USE IN NEWTON CLUSTER
        if args.use_newton_cluster:
            args.data_path = '/home/gabrieles/ears/code/data/rir/rir_indoor_4_channels_5.0_5.0_order_1_0.5_d_0.05/rir_indoor/'
        # USE IN FLORIA
        elif args.use_floria_cluster:
            # args.data_path = '/home/gabriele/EARS_project/data/rir/rir_indoor_4_channels_5.0_5.0_order_1_0.5_d_0.05/rir_indoor/'
            #args.data_path = '/mnt/walkure_public/tamirs/rir2d/rir_indoor/'

            # UNCOMMENT TO TO TRAIN ROBUSTLY AGAINST ORIENTATION
            # args.data_path = '/mnt/walkure_public/tamirs/rir_orientation_angle/rir_indoor_4_channels_orientation_5.0_5.0_order_1_0.5_d_0.05_d_angle_0.39269908169872414/rir_indoor/'

            #args.data_path = '/mnt/walkure_public/tamirs/pressure_field_2d_new_forward/indoor_recordings_4_rotors_8_mics_d_0.05_mode_indoor_None/'
            #args.data_path = '/mnt/walkure_public/tamirs/pressure_field_2d_circular_boundary/indoor_recordings_4_rotors_8_mics_d_0.05_mode_indoor_None/'
            args.data_path = '/mnt/walkure_public/tamirs/pressure_field_2d_no_padding/indoor_recordings_4_rotors_8_mics_d_0.05_mode_indoor_None/'
        else:
            raise ValueError(f"You need to set either --use-newton-cluster or --use-floria-cluster")
    elif args.use_orientation and args.data_path == '/home/tamir.shor/EARS/data/merged.npy':
        # args.data_path = '/home/gabrieles/ears/code/data/rir/rir_indoor_4_channels_orientation_5.0_5.0_order_1_0.5_d_0.05_d_angle_0.39269908169872414/rir_indoor/'
        args.data_path = '/datasets/rir_indoor'
    
    if args.duration_revolutions is not None:
        args.duration = (args.duration_revolutions*SAMPLES_PER_ROTATION)*PLOT_DT
    train(args)


def check_args(args):
    # correctness of noise injection arguments
    assert not (args.inject_noise_in_sound and (args.desired_snr_in_db is None) and (
                args.desired_snr_in_db_list is None)), "if inject_noise_in_sound is set, desired_snr_in_db or desired_snr_in_db_list must be set"
    assert not ((args.desired_snr_in_db is not None) and (
                args.desired_snr_in_db_list is not None)), "desired_snr_in_db and desired_snr_in_db_list are mutually exclusive"

    # correctness of window type for integrated velocity (in particular for the gaussian window type)
    assert not (
                args.fwhm is not None and args.window_type != 'gaussian'), "fwhm can be only specified for the gaussian window type. Please set --window-type to 'gaussian' or do not set fwhm"
    assert not (
                args.window_type == 'gaussian' and args.fwhm is None), "if you set --window-type to 'gaussian' you need to specify at least one value for the argument fwhm"

    assert not (args.duration is not None and args.duration_revolutions is not None), "Please choose only one between --duration and --duration-revolutions"

    assert (args.phase_modulation_snr_db_list is not None and args.inject_noise) or (not args.inject_noise), "You can't set --phase-modulation-snr-db-list without also setting --inject-noise and viceversa"

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int, help='Seed for random number generators')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')
    parser.add_argument('--data-path', type=str,
                        default='/home/tamir.shor/EARS/data/merged.npy', help='Path to the dataset')
    parser.add_argument('--sample-rate', type=float, default=1.,
                        help='Fraction of total volumes to include')
    parser.add_argument('--test-name', type=str, default='test/', help='name for the output dir')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='output/',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str, help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--report-interval', type=int, default=10, help='Period of loss reporting')

    # model parameters
    parser.add_argument('--num-layers', type=int, default=3, help='Number of Transformer layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--hidden-dim', type=int, default=1024, help='Transformer Hidden Dim')
    parser.add_argument('--num-heads', type=int, default=1, help='Transformer Number of heads')
    parser.add_argument('--data-parallel', action='store_true', default=False,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--decimation-rate', default=10, type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for each volume.')
    parser.add_argument('--avg-pool', action='store_true', default=False,
                        help='If set to True after the application of the Transformer, the output is averaged over the time dimension; alternatively, the model uses only the last token (default:False)')

    # optimization parameters
    parser.add_argument('--batch-size', default=4, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for reconstruction model')
    parser.add_argument('--lr-step-size', type=int, default=100,
                        help='Period of learning rate decay for reconstruction model')
    parser.add_argument('--lr-gamma', type=float, default=1,
                        help='Multiplicative factor of reconstruction model learning rate decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Optimizer Momentum')
    # learning parameters
    parser.add_argument('--sub-lr', type=float, default=0.005, help='learning rate of the phase layer')
    parser.add_argument('--sub-lr-time', type=float, default=1000,
                        help='learning rate decay timestep of the phase layer')
    parser.add_argument('--sub-lr-stepsize', type=float, default=1,
                        help='learning rate decay step size of the phase layer')

    parser.add_argument('--no-phase-shift-learn', action='store_false',
                        help='learn phase shifts. If set to False.')

    # MRI Machine Parameters
    parser.add_argument('--acc-weight', type=float, default=0.1, help='weight of the acceleration loss')
    parser.add_argument('--vel-weight', type=float, default=0.1, help='weight of the velocity loss')
    parser.add_argument('--rec-weight', type=float, default=1, help='weight of the reconstruction loss')

    parser.add_argument('--orientation-weight', type=float, default=1.0, help='weight of the orientation loss')

    parser.add_argument('--dt', type=float, default=1e-5, help='sampling time - sec')
    parser.add_argument('--a-max', type=float, default=4000, help='maximum acceleration')
    parser.add_argument('--v-max', type=float, default=8000, help='maximum velocity')
    parser.add_argument('--initialization', type=str, default='constant',
                        help='Initial Bezier Curve Shape')
    parser.add_argument('--use-closed-form-projection', action='store_true', default=False,
                        help='Project the phase modulation to the feasible set using a closed form solution.')

    # modelization parameters
    parser.add_argument('--interp-gap', type=int, default=0,
                        help='number of interpolated points between 2 parameter points in the phase curve')
    parser.add_argument('--num-rotors', type=int, default=4, help='number of drone rotors')
    parser.add_argument('--num-mics', type=int, default=8, help='number of microphones')
    parser.add_argument('--project', action='store_true', default=False,
                        help='Use projection to impose kinematic constraints.'
                             'If false, use interpolation and penalty (original PILOT paper).')
    parser.add_argument('--proj_iters', default=10e1, help='Number of iterations for each projection run.')
    parser.add_argument('--noise', action='store_true', default=False, help='add noise to phases.')

    # implementation parameters
    parser.add_argument('--no-fourier', action='store_true', default=False,
                        help='If set to True it does not compute the convolution between the input sound and the rir in the Fourier domain, but rather in the time domain')

    # freeze parameters
    parser.add_argument('--freeze-iter', type=int, default=1,
                        help='Iteration\'s frequency with which the trajectory is updated')
    parser.add_argument('--freeze-epoch', type=int, default=1,
                        help='Epoch\'s frequency with which the trajectory is updated')
    parser.add_argument('--gap-factor', type=float, default=0.0,
                        help='Interpolation Gap multiplication factor')
    parser.add_argument('--joint-epoch', type=int, default=-1,
                        help='Epoch from which to ignore freezes and optimize jointly.Set to -1 to ignore')
    parser.add_argument('--only-recons-epoch', type=int, default=10e6,
                        help='From Which Epoch to do reconstruction only. Set to -1 to ignore')
                        
    parser.add_argument('--warmup-epoch', type=int, default=-1,
                        help='Num of epochs to not do phase learning at start')
    parser.add_argument('--recons-reset-epoch-freq', type=int, default=-1,
                        help='Per how many epochs to do reconstruction reset. Set to -1 to ignore')
    parser.add_argument('--split-train', action='store_true', default=False,
                        help='Whether to freeze reconstruction when learning phases')
    parser.add_argument('--ignore-resets', type=int, default=-1,
                        help='Epoch from which to ignore reconstruction resets and optimize fully.Set to -1 to ignore')
    parser.add_argument('--use-l1', action='store_true', default=False, help='Use L1 loss')

    parser.add_argument('--inject-noise', action='store_true', default=False,
                        help='Whether to inject noise to the phase modulation used to create the modulated sound (the reconstruction model will still receive the uncorrupted phase)')

    parser.add_argument('--standardize', action='store_true', default=False,
                        help='Whether to standardize the input sound (mean 0, std 1) over the time domain')

    # 2d flags
    parser.add_argument('--use-2d', action='store_true', default=False, help='Whether to use 2d data')

    parser.add_argument('--use-cnn', action='store_true', default=False,
                        help='Whether to use CNNs for the reconstruction model')
    parser.add_argument('--use-wide-cnn', action='store_true', default=False,
                        help='Whether to use wide CNNs for the reconstruction model')

    parser.add_argument('--separate-training-full', action='store_true', default=False,
                        help='In separate training, whether to avoid learning backward params when optimizing phase')
    parser.add_argument('--separate-training', action='store_true', default=False,
                        help='Whether to train the phase and the reconstruction separately')
    parser.add_argument('--separate-epoch', type=int, default=10, help='Epochs dedicated separately to each model')
    parser.add_argument('--separate-epoch-reconstruction', type=int, default=5,
                        help='Epochs dedicated separately to each model')

    parser.add_argument('--separate-training-iters', action='store_true', default=False,
                        help='Whether to train the phase and the reconstruction separately')
    parser.add_argument('--separate-iter', type=int, default=10, help='Epochs dedicated separately to each model')
    parser.add_argument('--separate-iter-reconstruction', type=int, default=3,
                        help='Epochs dedicated separately to each model')

    # 2d + orientation flags
    parser.add_argument('--use-orientation', action='store_true', default=False,
                        help='Whether to use 2d+orientation data')

    parser.add_argument('--use-subset', action='store_true', default=False,
                        help="Whether to use a subset of the entire dataset (works only for orientation)")

    # noise injection in sound
    parser.add_argument('--inject-noise-in-sound', action='store_true', default=False,
                        help="Whether to inject noise in the sound (ATTENTION: you need to specify the variance of the noise in the argument --desired-snr-in-db)")
    parser.add_argument('--desired-snr-in-db', type=float, default=None,
                        help="Desired SNR given by the white gaussian noise injected in the sound measured in dB (ATTENTION: you need to specify the argument --inject-noise-in-sound)")
    parser.add_argument('--desired-snr-in-db-list', type=float, nargs="*",
                        help="Desired SNR values chosen uniformly random given by the white gaussian noise injected in the sound measured in dB (ATTENTION: you need to specify the argument --inject-noise-in-sound)")

    # integrated velocity
    parser.add_argument('--integrated-velocity-window-size', type=float, default=1.0,
                        help="Window size for the integrated velocity (in number of revolutions)")
    parser.add_argument('--v-integrated-max', type=float, default=1_000,
                        help='maximum integrated velocity in degrees per revolution')
    parser.add_argument('--v-integrated-weight', type=float, default=0.1, help='weight of the integrated velocity loss')
    parser.add_argument('--amp_penalty_coeff', type=float, default=0, help='amplitude penalty weight')
    parser.add_argument('--window-type', choices=['gaussian', 'ones'], type=str, default='gaussian',
                        help="Type of window used to integrate the velocity of the phase (choose between 'gaussian' and 'ones')")
    parser.add_argument('--fwhm', type=float, default=[0.7, 1.6, 2.7, 4.2], nargs='*',
                        help="List of FWHM in number of revolutions (attention: it can only be used with a gaussian)")

    # phase modeling
    parser.add_argument('--fourier-basis-size', type=int, default=10, help='Number of functions in Fourier Basis')
    parser.add_argument('--fourier-cutoff-freq', type=float, default=1,
                        help='Cutoff (lowest) frequency for Fourier basis')
    parser.add_argument('--use-fourier', action='store_true', default=True,
                        help="Whether to use Fourier Basis Modeling for phases")
    parser.add_argument('--use-cosine-basis', action='store_true', default=False,
                        help='Whether to use the cosine as a basis. If false it uses sine')

    parser.add_argument('--high-resolution-stft', action='store_true', default=True,
                        help='Whether to use a high resolution Short Time Fourier Transform in the preprocessing of the backward model')

    # cluster depedant flag
    parser.add_argument('--use-newton-cluster', action='store_true', default=False,
                        help="Whether to use Newton cluster")
    parser.add_argument('--use-floria-cluster', action='store_true', default=False, help="Whether to use Floria cluster")

    # train mode
    parser.add_argument('--validate', action='store_true', default=False, help="perform training or validation")

    parser.add_argument('--duration', type=float, default=None, help='Duration of the phase modulation in seconds')
    parser.add_argument('--duration-revolutions', type=float, default=None, help='Duration of the phase modulation in seconds')

    parser.add_argument('--sampling-interpolation-method', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='Which method to use to interpolate between the pressure field values (choose among bilinear, bicubic)')

    parser.add_argument('--phase-modulation-snr-db-list', type=float, nargs="*", default=[25., 30., 35.], help='SNR values in dB to choose from in order to corrupt the injected phase modulation with white Gaussian noise (in order to to inject this kind of noise set --inject-noise)')

    parser.add_argument('--local-testing', action='store_true', default=False)
    parser.add_argument('--init-size', type=float, default=1, help='Amplitude for initialization')

    parser.add_argument('--use-aggregate', action='store_true', default=False, help='Whether to use AggregateTransformer')
    parser.add_argument('--use-sound-transformer', action='store_true', default=False, help='Whether to use SoundTransformer')

    parser.add_argument('--use-sgd', action='store_true', default=False, help='Whether to use SGD optimizer')

    parser.add_argument('--use-2d-given-orientation', action='store_true', default=True, help='Whether to use 2d data')

    parser.add_argument('--use-orientation-transformer', action='store_true', default=False, help='Whether to use OrientationSpectrogramTransformer')
    parser.add_argument('--simulate-system', action='store_true', default=False, help='Whether to simulate the response of a system controlled by a PID controller')

    parser.add_argument('--use-orientation-aggregate-mlp', action='store_true', default=False, help='Whether to use OrientationAggregateTransformerMLP')
    parser.add_argument('--use-orientation-aggregate-attention', action='store_true', default=False, help='Whether to use OrientationAggregateTransformerAttention')

    parser.add_argument('--use-big-dataset', action='store_true', default=False, help='Whether to use the dataset with 32 angles per position')
    parser.add_argument('--use-mega-dataset', action='store_true', default=False, help='Whether to use the dataset with at most 128 angles per position')
    parser.add_argument('--use-orig-64-angles', action='store_true', default=False, help='Whether to use the dataset with at most 64 angles per position')

    parser.add_argument('--use-multi-position', action='store_true', default=False, help='Whether to use multiple recordings to estimate each coordinate')
    parser.add_argument('--position-number', type=int, default=2, help='Choose how many positions to use per coordinate (ignored when --trajectory-factory is set to all as it is by default)')

    parser.add_argument('--aggregator', choices=['deep', 'geometric_median', 'mean', 'transformer', 'mlp'], default='geometric_median', help='Which aggregator to use (choose among deep, geometric_median, mean, transformer, mlp)')

    parser.add_argument('--test-non-aggregating-trained', action='store_true', default=False, help='Whether to test a model that has not been trained with aggregation on aggregation')

    parser.add_argument('--system-noise-snr-db', type=float, default=float('inf'), help='Set the std of the white noise injected downstream of the PID controller response')
    
    parser.add_argument('--use-non-convex-room', action='store_true', default=False, help='Whether to use the non convex room')

    #parser.add_argument('--use-fast-system', action='store_true', default=True, help='Whether to use the fast system')
    parser.add_argument('--simulate-system-method', choices=["original", "time", "fourier"], default="fourier")

    parser.add_argument('--finetune', action='store_true', default=False, help='Whether to finetune a model')
    parser.add_argument('--trajectory-factory', choices=['ccw', 'all'], default='all', help='Which trajectory factory to use when --use-multi-position is set')

    parser.add_argument('--lr-aggregator', type=float, default=1e-5, help='Learning rate for the aggregator model')

    parser.add_argument('--use-asymmetric-non-convex-room', action='store_true', default=False, help='Whether to use the asymmetric non convex room dataset')

    parser.add_argument('--aggregator-dim-feedforward', type=int, default=2_048, help='Deep Aggregator feedforward dimension')
    parser.add_argument('--aggregator-num-heads', type=int, default=8, help='Deep Aggregator number of heads')
    parser.add_argument('--aggregator-dropout', type=float, default=0.2, help='Deep Aggregator dropout')
    parser.add_argument('--aggregator-num-layers', type=int, default=3, help='Deep Aggregator number of layers')

    return parser


if __name__ == '__main__':
    run()
