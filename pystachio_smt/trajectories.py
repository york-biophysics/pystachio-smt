#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

""" TRAJECTORIES - Trajectory construction and manipulation module

Description:
    trajectories.py contains the Trajectory class containing information about
    spot trajectories across multiple frames, along with the routines involved
    in locating and extending the trajectories.

Contains:
    class Trajectory
    function build_trajectories
    function read_trajectories
    function write_trajectories

Author:
    Edward Higgins

Version: 0.2.0
"""

import csv
import sys
import os

import numpy as np
from spots import Spots


class Trajectory:
    def __init__(self, id, spots, spot_id):
        self.id = id
        self.start_frame = spots.frame
        self.end_frame = spots.frame
        self.path = [spots.positions[spot_id, :]]
        self.intensity = [spots.spot_intensity[spot_id]]
        self.bg_intensity = [spots.bg_intensity[spot_id]]
        self.snr = [spots.snr[spot_id]]
        self.length = 1
        self.stoichiometry = 0
        self.converged = [spots.converged[spot_id]]
        self.linked_traj = None
        self.width = [spots.width[spot_id]]
        # Lewis edits
        self.noise = [spots.noise[spot_id]]
        self.precision = [spots.precision[spot_id]]

    def extend(self, spots, spot_id):
        if spots.frame > self.end_frame + 1:
            sys.exit("ERROR: Cannot extend a spot over multiple frames")

        self.end_frame = spots.frame
        self.path.append(spots.positions[spot_id, :])
        self.intensity.append(spots.spot_intensity[spot_id])
        self.bg_intensity.append(spots.bg_intensity[spot_id])
        self.converged.append(spots.converged[spot_id])
        self.snr.append(spots.snr[spot_id])
        self.width.append(spots.width[spot_id])
        self.noise.append(spots.noise[spot_id])
        self.precision.append(spots.precision[spot_id])

        self.length += 1

def build_trajectories(all_spots, params):
    trajectories = []
    traj_num = 0

    # Create a trajectory for all spots in the first frame
    for i in range(all_spots[0].num_spots):
        trajectories.append(Trajectory(traj_num, all_spots[0], i))
        traj_num += 1

    # Construct trajectories for the rest of the frames
    for frame in range(1, len(all_spots)):
        assigned_spots = []
        for spot in range(all_spots[frame].num_spots):
            close_candidates = []
            for candidate in trajectories:
                if candidate.end_frame != frame - 1:
                    continue
                candidate_dist = np.linalg.norm(
                    all_spots[frame].positions[spot, :] - candidate.path[-1]
                )
                if candidate_dist < params.max_displacement:
                    close_candidates.append(candidate)

            if len(close_candidates) == 0:
                trajectories.append(Trajectory(traj_num, all_spots[frame], spot))
                traj_num += 1

            elif len(close_candidates) == 1:

                close_candidates[0].extend(all_spots[frame], spot)

            else:
                trajectories.append(Trajectory(traj_num, all_spots[frame], spot))
                traj_num += 1

    filtered_trajectories = list(filter(lambda x: x.length >= params.min_traj_len, trajectories))

    actual_traj_num = 0
    for traj in filtered_trajectories:
        traj.id = actual_traj_num
        actual_traj_num += 1

    return filtered_trajectories


def write_trajectories(trajectories, filename): # Lewis correct width from traj.width[0][x/y] to traj.width[i][x/y] and added noise + precision
    f = open(filename, "w")
    f.write(f"trajectory\tframe\tx\ty\tspot_intensity\tbg_intensity\tSNR\tconverged\twidthx\twidthy\tnoise\tprecision_x\tprecision_y\n")
    for traj in trajectories:
        for frame in range(traj.start_frame, traj.end_frame + 1):
            i = frame - traj.start_frame

            f.write(
                f"{traj.id}\t{frame}\t{traj.path[i][0]}\t{traj.path[i][1]}\t{traj.intensity[i]}\t{traj.bg_intensity[i]}\t{traj.snr[i]}\t{traj.converged[i]}\t{traj.width[i][0]}\t{traj.width[i][1]}\t{traj.noise[i]}\t{traj.precision[i][0]}\t{traj.precision[i][1]}\n"
            )
    f.close()


def to_spots(trajs):
    all_spots = []
    frame = 0
    done_all_frames = False
    while not done_all_frames:
        done_all_frames = True

        # Get the spots from trajs for this frame
        positions = []
        bg_intensity = []
        spot_intensity = []
        snr = []
        converged = []
        width = []
        noise = []
        precision = []
        for traj in trajs:

            if traj.end_frame > frame:
                done_all_frames = False

            if traj.start_frame > frame or traj.end_frame < frame:
                continue

            i = frame-traj.start_frame
            positions.append(traj.path[i][:])
            spot_intensity.append(traj.intensity[i])
            bg_intensity.append(traj.bg_intensity[i])
            snr.append(traj.snr[i])
            converged.append(traj.converged[i])
            width.append(traj.width[i]) # Lewis changed to width[i] from width
            # Lewis
            noise.append(traj.noise[i])
            precision.append(traj.precision[i])


        spots = Spots(len(positions), frame)
        spots.set_positions(np.array(positions))
        spots.spot_intensity = np.array(spot_intensity)
        spots.bg_intensity = np.array(bg_intensity)
        spots.snr = np.array(snr)
        spots.converged = np.array(converged, dtype=np.int8)
        spots.width=np.array(width)
        #Lewis
        spots.noise = np.array(noise)
        spots.precision = np.array(precision)
        
        all_spots.append(spots)

        frame += 1

    return all_spots

def read_trajectories(filename):
    trajectories = []
    prev_traj_id = -1
    if not os.path.isfile(filename):
        print(f"WARNING: No such file {filename}")
        return None
    with open(filename) as tsv_file:
        tsv_reader = csv.reader(tsv_file,delimiter="\t")
        for line in tsv_reader:
            if line[0] == "trajectory":
                continue
            spot = Spots(num_spots=1)
            traj_id = int(line[0])
            spot.frame = int(line[1])
            spot.positions[0, :] = [float(line[2]), float(line[3])]
            spot.spot_intensity[0] = float(line[4])
            spot.bg_intensity[0] = float(line[5])
            spot.snr[0] = float(line[6])
            spot.converged[0] = int(line[7])
            spot.width[0,:] = [float(line[8]),float(line[9])]
            # Lewis
            spot.noise[0] = float(line[10])
            spot.precision[0,:] = [float(line[11]),float(line[12])]

            if traj_id != prev_traj_id:
                trajectories.append(Trajectory(traj_id, spot, 0))
            else:
                trajectories[-1].extend(spot, 0)
            prev_traj_id = traj_id

    return trajectories

def compare_trajectories(params):
    all_target_spots = []
    all_spots = []

    trajs = read_trajectories(params.name + "_trajectories.tsv")
    target_trajs = read_trajectories(params.name + "_simulated.tsv")

    frame = 0
    done_all_frames = False
    while not done_all_frames:
        done_all_frames = True

        # Get the spots from target_trajs for this frame
        frame_target_spots = []
        for traj in target_trajs:

            if traj.end_frame > frame:
                done_all_frames = False

            if traj.start_frame > frame or traj.end_frame < frame:
                continue
            frame_target_spots.append(traj.path[frame-traj.start_frame][:])

        target_spots = Spots(len(frame_target_spots), frame)
        target_spots.set_positions(np.array(frame_target_spots))
        all_target_spots.append(target_spots)

        # Get the spots from trajs for this frame
        frame_spots = []
        for traj in trajs:

            if traj.end_frame > frame:
                done_all_frames = False

            if traj.start_frame > frame or traj.end_frame < frame:
                continue

            frame_spots.append(traj.path[frame-traj.start_frame][:])

        spots = Spots(len(frame_spots), frame)
        spots.set_positions(np.array(frame_spots))
        all_spots.append(spots)

        frame += 1

    num_frames = frame

    for frame in range(num_frames):
        false_positives = 0
        false_negatives = 0
        multiples = 0
        matches = 0
        errors = []
        outside_frame = 0

        assigned_spots = []
        for spot in range(all_target_spots[frame].num_spots):

            if all_target_spots[frame].positions[spot,0] < 0 \
                      or all_target_spots[frame].positions[spot,1] < 0 \
                      or all_target_spots[frame].positions[spot,0] >= params.frame_size[0] \
                      or all_target_spots[frame].positions[spot,1] >= params.frame_size[1]:
              outside_frame  += 1
              continue

            close_candidates = []
            for candidate in range(all_spots[frame].num_spots):
                candidate_dist = np.linalg.norm(
                    all_target_spots[frame].positions[spot,:] - all_spots[frame].positions[candidate,:]
                )
                if candidate_dist < 1:
                    close_candidates.append(candidate_dist)

            if len(close_candidates) == 0:
                false_negatives += 1

            elif len(close_candidates) == 1:
                matches += 1
                errors.append(close_candidates[0])

            else:
                false_negatives += 1
                multiples += 1

        false_positives = all_spots[frame].num_spots - matches
        print(f"Frame {frame:4d}:",
              f"Error = {np.mean(np.array(errors)):8f} pixels, ",
              f"Found = {all_spots[frame].num_spots:3d} ",
              f"Matches = {matches:3d} ",
              f"False negatives = {false_negatives:3d} ",
              f"False positives = {false_positives:3d} ",
              f"Outside = {outside_frame:3d} ")
