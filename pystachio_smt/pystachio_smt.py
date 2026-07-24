#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

""" SMT - Single Molecule Tools program

Description:
    SMT.py contains the main program used for running SMT-Python

Contains:
    function main

Author:
    Edward Higgins

Version: 0.2.0
"""

import sys
import numpy as np

import images
import parameters
import postprocessing
import simulation
import tracking
import trajectories
import visualisation
import dash_ui.launcher
import preprocess
import os
import datetime

def save_parameters(params, save_dir="."):
    
    # 1. Define parameter categories
    PARAM_GROUPS = {
        "General": [
            "task", "name", "overwrite", "num_frames", "pixel_size", 
            "use_channel", "num_channels", "ALEX"
        ],
        "Preprocessing": [
            "video_path", "bead_path", "bf_path", "tmats_path", "roi_file", 
            "roi_channel", "mask_type", "mask_prefix", "model", "model_type", 
            "area_filter", "cell_fitting", "inv_bf", "use_otsu", "frame_avg",
            "manual_registration", "manual_pairs"
        ],
        "Tracking": [
            "search_radius", "min_trajectory_len", "max_linking_distance", 
            "tracking_method", "detection_threshold"
        ],
        "Simulation / Postprocessing": [
            "sim_type", "postprocess_mode", "output_format"
        ]
    }

    # 2. Extract key-value dictionary safely
    if hasattr(params, '__dict__'):
        param_dict = vars(params)
    else:
        param_dict = {
            attr: getattr(params, attr) 
            for attr in dir(params) 
            if not attr.startswith('_') and not callable(getattr(params, attr))
        }

    # 3. Clean string values
    cleaned_params = {}
    for k, v in param_dict.items():
        if callable(v):
            continue
        if isinstance(v, (list, tuple, set)):
            cleaned_params[k] = ", ".join(map(str, v)) if v else "None"
        elif v is None or str(v).strip() == "":
            cleaned_params[k] = "None"
        else:
            cleaned_params[k] = str(v)

    # 4. Calculate longest key length across all keys for uniform alignment
    max_key_len = max((len(k) for k in cleaned_params.keys()), default=20)

    # 5. Track which keys have been categorized
    written_keys = set()
    filepath = os.path.join(save_dir, "parameters.txt")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write(" PySTACHIO Parameters\n")
        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("==================================================\n")

        # Write categorized sections
        for group_name, keys in PARAM_GROUPS.items():
            # Check if any key from this group exists in the parameters
            present_keys = [k for k in keys if k in cleaned_params]
            if not present_keys:
                continue

            f.write(f"\n--- [{group_name}] ---\n")
            for k in sorted(present_keys):
                f.write(f"{k:<{max_key_len}} : {cleaned_params[k]}\n")
                written_keys.add(k)

        # Catch-all section for any parameters not explicitly in PARAM_GROUPS
        remaining_keys = set(cleaned_params.keys()) - written_keys
        if remaining_keys:
            f.write("\n--- [Other / Custom] ---\n")
            for k in sorted(remaining_keys):
                f.write(f"{k:<{max_key_len}} : {cleaned_params[k]}\n")

        f.write("\n==================================================\n")

    print(f"Parameters saved to: {filepath}", flush=True)

def main():
    args = sys.argv[1:]
    
    if "-r" in args:
        try:
            # Find the -r flag and extract the filename that follows it
            r_index = args.index("-r")
            config_file = args.pop(r_index + 1)
            args.pop(r_index) # Remove the '-r' flag itself
            
            with open(config_file, 'r', encoding='utf-8') as f:
                file_args = []
                for line in f:
                    # Ignore anything after a '#' (comments) and strip whitespace
                    clean_line = line.split('#')[0].strip()
                    if clean_line:
                        file_args.append(clean_line)
            
            args = file_args + args
            
        except IndexError:
            sys.exit("ERROR: -r flag provided but no config file specified.")
        except FileNotFoundError:
            sys.exit("ERROR: Config file not found.")
    # ---------------------------------

    params = parameters.Parameters()
    params.read(args)
    
    save_parameters(params, save_dir=".")
    
    sim=False
    
    for task in params.task:
        if task == "app":
            dash_ui.launcher.launch_app(params)

        elif task == "help":
            # Safely check if a specific help topic was requested
            if len(args) > 0:
                params.help(args[0])
            else:
                params.help()

        elif task == "preprocess":
            preprocess.run_preprocessing(params)

        elif task == "track":
            tracking.track(params)

        elif task == "simulate":
            simulation.simulate(params)
            sim=True

        elif task == "postprocess":
            postprocessing.postprocess(params, simulated=sim)

        elif task == "view":
            visualisation.render(params)

        else:
            sys.exit(f"ERROR: Task {task} is not yet implemented. Aborting...")

if __name__ == "__main__":
    main()

