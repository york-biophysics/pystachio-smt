#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

""" SIMULATION - Dataset simulation module

Description:
    simulation.py contains the code for the simulation task, which simulates
    pseudo-experimental datasets as characterised by the relevant parameters.

Contains:
    function simulate

Author:
    Edward Higgins

Version: 0.2.0
"""

from functools import reduce

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import sys

from images import ImageData
from spots import Spots
import trajectories


def simulate(params):
    if params.num_frames < 1:
        sys.exit("ERROR: Cannot simulate image with num_frames < 0")

    # Make a spot array the same size as normal
    real_spots = [Spots(params.num_spots) for i in range(params.num_frames)]
    if params.max_spot_molecules == 1:
        n_mols = np.array([1] * params.num_spots)
    else:
        n_mols = np.array(random.randint(1, params.max_spot_molecules, params.num_spots))
    n_mols_fractional_intensity = np.zeros(n_mols.shape)

    # initialise the spot co-ords
    real_spots[0].positions[:, 0] = random.rand(params.num_spots) * params.frame_size[0]
    real_spots[0].positions[:, 1] = random.rand(params.num_spots) * params.frame_size[1]
    real_spots[0].spot_intensity[:] = params.I_single * (n_mols+n_mols_fractional_intensity)
    real_spots[0].frame = 1

    # Simulate diffusion
    S = np.sqrt(2 * params.diffusion_coeff * params.frame_time) / params.pixel_size
    frame_start = 0
    frame_end = params.num_spots

    for frame in range(1, params.num_frames):
        real_spots[frame].frame = frame
        real_spots[frame].spot_intensity[:] = params.I_single * (n_mols+n_mols_fractional_intensity)
        real_spots[frame].traj_num = real_spots[frame - 1].traj_num[:]
        real_spots[frame].positions = random.normal(
            real_spots[frame - 1].positions, S, (params.num_spots, 2)
        )

        # Photobleah some spots
        n_mols_fractional_intensity[:] = 0
        for i in range(params.num_spots):
            if n_mols[i] > 0:
                for j in range(n_mols[i]):
                    if random.rand() < params.p_bleach_per_frame:
                        #How far into next frame does this one last?
                        frac = random.rand()
                        n_mols_fractional_intensity[i] += frac
                        n_mols[i] -= 1
            if n_mols[i] == 0 and params.photoblink:
                if random.rand() < params.p_photoblink:
                    n_mols[i] = 1

    # Simulate the image stack and save
    image = ImageData()
    image.initialise(params.num_frames, params.frame_size)

    x_pos, y_pos = np.meshgrid(range(params.frame_size[0]), range(params.frame_size[1]))
    for frame in range(params.num_frames):
        frame_data = np.zeros([params.frame_size[1], params.frame_size[0]]).astype(np.uint16)

        for spot in range(params.num_spots):
            spot_data = (
                (real_spots[frame].spot_intensity[spot] / (2 * np.pi * params.spot_width**2))
                * np.exp(
                    -(
                        (x_pos - real_spots[frame].positions[spot, 0]) ** 2
                        + (y_pos - real_spots[frame].positions[spot, 1]) ** 2
                    )
                    / (2 * params.spot_width ** 2)
                )
            ).astype(np.uint16)
            frame_data += spot_data
            real_spots[frame].spot_intensity[spot]=np.sum(spot_data)

        frame_data = random.poisson(frame_data)
        bg_noise = random.normal(params.bg_mean, params.bg_std, [params.frame_size[1], params.frame_size[0]])
        frame_data += np.where(bg_noise > 0, bg_noise.astype(np.uint16), 0)
        image[frame] = frame_data

    real_trajs = trajectories.build_trajectories(real_spots, params)

    image.write(params.name + ".tif")
    trajectories.write_trajectories(real_trajs, params.name + '_simulated.tsv')
    return image, real_trajs
