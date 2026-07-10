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

