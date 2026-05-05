#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.

""" SIMULATION - Dataset simulation module

Description:
    simulation.py contains the code for the simulation task, which simulates
    pseudo-experimental datasets as characterised by the relevant parameters.

Contains:
    function simulate
    function simulate_spherical_volume
    function simulate_spherical_surface

Authors:
    Jack Shepherd and Ed Higgins

Version: 1.1
"""

from functools import reduce

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sys
import tifffile

from images import ImageData
from spots import Spots
import trajectories

from numpy.random import Generator, PCG64
rng = Generator(PCG64())

def simulate(params):
    if params.num_frames < 1:
        sys.exit("ERROR: Cannot simulate image with num_frames < 0")

    # Make a spot array the same size as normal
    real_spots = [Spots(params.num_spots) for i in range(params.num_frames)]
    if params.max_spot_molecules == 1:
        n_mols = np.array([1] * params.num_spots)
    else:
        n_mols = np.array(rng.integers(1, params.max_spot_molecules, params.num_spots, endpoint=True))
    n_mols_fractional_intensity = np.zeros(n_mols.shape)

    # initialise the spot co-ords
    real_spots[0].positions[:, 0] = rng.random(params.num_spots) * params.frame_size[0]
    real_spots[0].positions[:, 1] = rng.random(params.num_spots) * params.frame_size[1]
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
        real_spots[frame].positions = rng.normal(
            real_spots[frame - 1].positions, S, (params.num_spots, 2)
        )

        # Photobleah some spots
        n_mols_fractional_intensity[:] = 0
        for i in range(params.num_spots):
            if n_mols[i] > 0:
                for j in range(n_mols[i]):
                    if random.rand() < params.p_bleach_per_frame:
                        #How far into next frame does this one last?
                        frac = rng.random()
                        n_mols_fractional_intensity[i] += frac
                        n_mols[i] -= 1
            if n_mols[i] == 0 and params.photoblink:
                if rng.random() < params.p_photoblink:
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

        frame_data = rng.poisson(frame_data)
        bg_noise = rng.normal(params.bg_mean, params.bg_std, [params.frame_size[1], params.frame_size[0]])
        frame_data += np.where(bg_noise > 0, bg_noise.astype(np.uint16), 0)
        image[frame] = frame_data

    real_trajs = trajectories.build_trajectories(real_spots, params)

    image.write(params.name + ".tif")
    trajectories.write_trajectories(real_trajs, params.name + '_simulated.tsv')
    return image, real_trajs

def simulate_spherical_volume(params):
    if params.psf_name is None:
        if params.pixel_size != 0.05:
            print("WARNING: Default 3D PSF only works with pixel_size = 50 nm. Changing params.pixel_size to 50 nm")
            params.pixel_size = 0.050
        params.psf_name = '3d_psf.npy'
    psf = np.load(params.psf_name)
    r = params.spherical_volume_radius
    D = params.diffusion_coeff
    dt = params.frame_time
    sigma = np.sqrt(2*D*dt)
    output_image = np.zeros((params.num_frames, 101, 101))
    v = np.zeros((params.num_spots,3), dtype='float64')
    for i in range(params.num_spots):
        while True:
            tmp_ = rng.uniform(-r,r,3)
            if np.sum(tmp_**2) <= r**2:
                v[i,:] = np.copy(tmp_)
                break
    for i in range(params.num_frames):
        tmp_im = np.zeros((101,101,101)) # Central spot is 50,50,50, which we will take as the origin
        for particle in range(params.num_spots):
            dv = rng.normal(0,sigma,3)
            tmpv = v[particle,:]
            tmpv += dv
            if np.sum(tmpv**2) > r**2:
                a = dv[0]**2 + dv[1]**2 + dv[2]**2
                b = -2 * (tmpv[0]*dv[0] + tmpv[1]*dv[1] + tmpv[2]*dv[2])
                c = tmpv[0]**2 + tmpv[1]**2 + tmpv[2]**2 - r**2
                alpha1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a) 
                alpha2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
                if abs(alpha1)<abs(alpha2):
                    tmpv -= alpha1*dv
                else:
                    tmpv -= alpha2*dv
                # Check it worked
                if tmpv[0]**2 + tmpv[1]**2 + tmpv[2]**2 - r**2 > 1E-6:
                    print("ERROR: Didn't find surface!")
                    exit(1)
            v[particle,:] = np.copy(tmpv)
            xpix = int(v[particle,2]/params.pixel_size) + 50
            ypix = int(v[particle,1]/params.pixel_size) + 50
            zpix = int(v[particle,0]/params.pixel_size) + 50
            tmp_im[zpix-psf.shape[0]//2:zpix+psf.shape[0]//2+1,ypix-psf.shape[1]//2:ypix+psf.shape[1]//2+1,xpix-psf.shape[2]//2:xpix+psf.shape[2]//2+1] += psf[:,:,:]
        #Take sum of central slices to produce simulated field of view, assuming a 500 nm depth of field
        output_image[i,:,:] = np.sum(tmp_im[45:56,:,:], axis=0)
    tifffile.imwrite(params.name+'.tif', np.array(output_image))
    return 0

def simulate_spherical_surface(params):
    from algorithms import rotate_3d_vector
    if params.psf_name is None:
        if params.pixel_size != 0.05:
            print("WARNING: Default 3D PSF only works with pixel_size = 50 nm. Changing params.pixel_size to 50 nm")
            params.pixel_size = 0.050
        params.psf_name = '3d_psf.npy'
    psf = np.load(params.psf_name)
    r = params.spherical_volume_radius
    D = params.diffusion_coeff
    dt = params.frame_time
    sigma = np.sqrt(2*D*dt)
    output_image = np.zeros((params.num_frames, 101, 101))
    v = np.zeros((params.num_spots,3), dtype='float64')
    for i in range(params.num_frames):
        tmp_im = np.zeros((101,101,101)) # Central spot is 50,50,50, which we will take as the origin
        for particle in range(params.num_spots):
            d = np.random.normal(0,sigma,2)
            dr = np.sqrt(np.sum(d**2))
            # Generate a random axis
            while True:
                axis = np.array([np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1)])
                norm = np.linalg.norm(axis)
                if norm <= 1:
                    axis /= norm
                    break
            r_to_axis = np.linalg.norm(np.cross(v,axis))
            dtheta = dr/r_to_axis
            v = np.dot(rotation_matrix(axis,dtheta),v)
    return 0
