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
<<<<<<< Updated upstream
import visualisation
import dash_ui.launcher
=======
import preprocess
>>>>>>> Stashed changes

def main():
    params = parameters.Parameters()
    params.read(sys.argv[1:])
    sim=False
    
    for task in params.task:
        if task == "app":
            dash_ui.launcher.launch_app(params)

        elif task == "help":
            if (len(sys.argv) > 2):
                params.help(sys.argv[2])
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

        elif task == "compare":
            trajectories.compare_trajectories(params)

        else:
            sys.exit(f"ERROR: Task {task} is not yet implemented. Aborting...")


if __name__ == "__main__":
    main()

