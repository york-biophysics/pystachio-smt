#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

""" PARAMETERS - Program parameters module

Description:
    parameters.py contains the Parameters class that holds all the program
    parameters, along with the default value for each parameter and routines
    for setting those parameters.

Contains:
    class Parameters

Author:
    Edward Higgins

Version: 0.2.0
"""

import sys
from difflib import SequenceMatcher

default_parameters = {
    # Runtime parameters
    'display_figures':
        { 'description': 'Figures (Y/n)',
          'level': 'basic',
          'class': 'general',
          'default': True },    
    'num_procs':
        { 'description': 'The number of CPU processes to run with (0 is serial)',
          'level': 'basic',
          'class': 'general',
          'default': 0 },
    'tasks':
        { 'description': 'Which task(s) to perform in the run',
          'level': 'basic',
          'class': 'general',
          'default': [],
          'options': ['preprocess', 'simulate', 'track', 'postprocess', 'view', 'app'] }, # Added 'preprocess'
    'name':
        { 'description': 'Name prefixing all files associated with this run',
          'level': 'basic',
          'class': 'general',
          'default': '' },
    'mask_file':
        { 'description': 'Filename of an image mask for filtering spots',
          'level': 'basic',
          'class': 'general',
          'default':  ''},
    'verbose':
        { 'description': 'Whether or not to display progress messages',
          'level': 'basic',
          'class': 'general',
          'default':  True},

    # Image parameters
    'frame_time':
        { 'description': 'Time per frame in seconds',
          'level': 'advanced',
          'class': 'image',
          'default': 0.005 },
    'pixel_size':
        { 'description': 'Length of a single pixel in μm',
          'level': 'advanced',
          'class': 'image',
          'default': 0.120 },
    'psf_width':
        { 'description': '?',
          'level': 'advanced',
          'class': 'image',
          'default': 0.120 },
    'start_frame':
        { 'description': 'The first frame of the image stack to analyse',
          'level': 'basic',
          'class': 'image',
          'default':  0},
    'num_frames':
        { 'description': 'Number of frames to simulate/analyse',
          'level': 'basic',
          'class': 'image',
          'default': 0 },
    'use_channel':
        { 'description': 'Flag to use only one channel of a non-ALEX acquisition',
         'level': 'basic',
         'class': 'image',
         'options': ['None', 'Left', 'Right', 'Both'],
         'default': 'None' },
    'channel_split':
        { 'description': 'If/how the frames are split spatially',
          'level': 'basic',
          'class': 'image',
          'default': 'None',
          'options': ['None', 'Vertical', 'Horizontal'] },
    'cell_mask':
        { 'description': 'Name of a black/white TIF file containing a cell mask',
          'level': 'advanced',
          'class': 'image',
          'default': ''},
    'ALEX':
        { 'description': 'Perform Alternating-Laser experiment analysis',
          'level': 'basic',
          'class': 'image',
          'default': False},
    'start_channel':
        { 'description': '?',
          'level': 'basic',
          'class': 'image',
          'default': 'L'},

    # Simulation parameters
    'num_spots':
        { 'description': 'Number of spots to simulate',
          'level': 'basic',
          'class': 'simulation',
          'default': 10 },
    'frame_size':
        { 'description': 'Size of frame to simulate ([x,y])',
          'level': 'basic',
          'class': 'simulation',
          'default': [100,100] },
    'I_single':
        { 'description': 'I_single value for simulated spots',
          'level': 'basic',
          'class': 'simulation',
          'default': 10000.0 },
    'bg_mean':
        { 'description': 'Mean of the background pixlel intensity',
          'level': 'advanced',
          'class': 'simulation',
          'default': 500.0 },
    'bg_std':
        { 'description': 'Standard deviation of the background pixel intensity',
          'level': 'advanced',
          'class': 'simulation',
          'default': 120.0 },
    'diffusion_coeff':
        { 'description': 'Diffusion coefficient of the diffusing spots',
          'level': 'basic',
          'class': 'simulation',
          'default': 1.0 },
    'spot_width':
        { 'description': 'Width of the simulated Gaussian spot',
          'level': 'advanced',
          'class': 'simulation',
          'default':  1.33},
    'max_spot_molecules':
        { 'description': 'Maximum number of dye molecules per spot',
          'level': 'advanced',
          'class': 'simulation',
          'default': 1 },
    'p_bleach_per_frame':
        { 'description': 'Probability of a spot bleaching in a given frame',
          'level': 'advanced',
          'class': 'simulation',
          'default': 0.0 },
    'photobleach':
        { 'description': 'Perform photobleaching (alias for max_spot_molecules=10, p_bleach_per_frame=0.05)',
          'level': 'basic',
          'class': 'simulation',
          'default': False },

    # Tracking parameters
    'bw_threshold_tolerance':
        { 'description': 'Threshold for generating the b/w image relative to the peak intensity',
          'level': 'advanced',
          'class': 'tracking',
          'default': 1.0 },
    'snr_filter_cutoff':
        { 'description': 'Cutoff value when filtering spots by signal/noise ratio',
          'level': 'basic',
          'class': 'tracking',
          'default': 0.4 },
    'filter_image':
        { 'description': 'Method for filtering the input image pre-analysis',
          'level': 'advanced',
          'class': 'tracking',
          'default': 'Gaussian',
          'options': ['Gaussian', 'None']},
    'max_displacement':
        { 'description': 'Maximum displacement allowed for spots between frames',
          'level': 'advanced',
          'class': 'tracking',
          'default': 5.0 },
    'struct_disk_radius':
        { 'description': 'Radius of the Disk structural element',
          'level': 'advanced',
          'class': 'tracking',
          'default': 5 },
    'min_traj_len':
        { 'description': 'Minimum number of frames needed to define a trajectory',
          'level': 'advanced',
          'class': 'tracking',
          'default': 3 },
    'subarray_halfwidth':
        { 'description': 'Halfwidth of the sub-image for analysing individual spots',
          'level': 'advanced',
          'class': 'tracking',
          'default': 8 },
    'gauss_mask_sigma':
        { 'description': 'Width of the Gaussian used for the iterative centre refinement',
          'level': 'advanced',
          'class': 'tracking',
          'default': 2.0 },
    'gauss_mask_max_iter':
        { 'description': 'Max number of iterations for the iterative centre refinement',
          'level': 'advanced',
          'class': 'tracking',
          'default': 1000 },
    'inner_mask_radius':
        { 'description': 'Radius of the mask used for calculating spot intensities',
          'level': 'advanced',
          'class': 'tracking',
          'default': 5 },

    # Postprocessing parameters
    'display_figures':
        { 'description': 'Whether or not to display the figures live in MatPlotLib',
          'level': 'basic',
          'class': 'postprocessing',
          'default': True},
    'chung_kennedy_window':
        { 'description': 'Window width for Chung-Kennedy filtering',
          'level': 'basic',
          'class': 'postprocessing',
          'default': 3},
    'chung_kennedy':
        { 'description': 'Flag to specify whether or not to Chung-Kennedy filter intensity tracks',
          'level': 'basic',
          'class': 'postprocessing',
          'default': False},
    'msd_num_points':
        { 'description': 'Number of points used to calculate the mean-squared displacement',
          'level': 'basic',
          'class': 'postprocessing',
          'default': 4 },
    'stoic_method':
        { 'description': 'Method used for determining the stoichiometry of each trajectory',
          'level': 'advanced',
          'class': 'postprocessing',
          'default': 'Linear',
          'options': ['Linear', 'Mean', 'Initial'] },
    'num_stoic_frames': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': 'Number of frames used to determine the stoichiometry',
          'default': 3 },
    'calculate_isingle': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': 'Whether or not to calculate the ISingle',
          'default': True },
    'colocalize': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': '?',
          'default': False },
    'colocalize_distance': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': '?',
          'default': 5 },
    'colocalize_n_frames': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': '?',
          'default': 5 },
    'copy_number': {
          'level': 'advanced',
          'class': 'postprocessing',
          'description': '?',
          'default': False },
    'L_isingle':{
        'level' : 'intermediate',
        'class' : 'postprocessing',
        'default' : '10000',
        },
    'R_isingle':{
        'level' : 'intermediate',
        'class' : 'postprocessing',
        'default' : '10000',
        },
<<<<<<< Updated upstream
=======
    'isingle_fraction':{
        'level' : 'advanced',
        'class' : 'postprocessing',
        'description' : 'What % of frames to use when estimating Isingle. The specified percentage will be at the end of the acquisition, i.e. when things should have photobleached already',
        'default' : 66
        },
    
    # Preprocessing parameters
    'video_path':
        { 'description': 'Path to the raw video TIFF',
          'level': 'basic',
          'class': 'preprocessing',
          'default': '' },
    'bf_path':
        { 'description': 'Path to the brightfield image',
          'level': 'basic',
          'class': 'preprocessing',
          'default': '' },
    'bead_path':
        { 'description': 'Path to the calibration bead TIFF',
          'level': 'basic',
          'class': 'preprocessing',
          'default': '' },
    'manual_registration':
        { 'description': 'Set True to manually click beads for alignment',
          'level': 'basic',
          'class': 'preprocessing',
          'default': False },
    'manual_pairs':
        { 'description': 'Number of bead pairs to click if manual_registration is True',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': 10 },
    'tmats_path':
        { 'description': 'Path to save/load transformation matrix',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': './transformation_matrix.npy' },
    'model':
        { 'description': 'Path to the segmentation model file (.h5)',
          'level': 'basic',
          'class': 'preprocessing',
          'default': 'model.h5' },
    'model_type':
        { 'description': 'Type of model (e.g., unet)',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': 'unet' },
    'mask_type':
        { 'description': 'Type of mask to generate',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': 'cell' },
    'frame_avg':
        { 'description': 'Number of frames to average for ROI/Mask generation',
          'level': 'basic',
          'class': 'preprocessing',
          'default': 5 },
    'roi_channel':
        { 'description': 'Channel to use for ROI generation (L or R)',
          'level': 'basic',
          'class': 'preprocessing',
          'default': 'L' },
    'roi_file':
        { 'description': 'Path to the ImageJ ROI zip file',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': 'roi.zip' },
    'area_filter':
        { 'description': 'Minimum area size for cell filtering',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': 0 },
    'inv_bf':
        { 'description': 'Invert brightfield image intensity',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': False },
    'cell_fitting':
        { 'description': 'Perform precise cell boundary fitting',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': False },
    'mask_prefix':
        { 'description': 'Prefix for saved mask files',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': '' },
    'use_otsu':
        { 'description': 'Use Otsu thresholding instead of model segmentation',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': False },
    'overwrite':
        { 'description': 'Overwrite existing output files',
          'level': 'advanced',
          'class': 'preprocessing',
          'default': True }
>>>>>>> Stashed changes
}


class Parameters:
    def __init__(self, initial=default_parameters):
        self._params = initial

        for param in self._params.keys():
            # Set all the values to be the default values
            self._params[param]['value'] = self._params[param]['default']

#EJH#         self.num_procs = 0
#EJH#         self.verbose = True  # Whether or not to display verbose console output
#EJH#         self.c_split = "None"  # How the channels are split
#EJH#         self.frames_to_track = (
#EJH#             0  # How many frames to track after the laser has switched on
#EJH#         )
#EJH#         self.start_channel = 0  # First channel to use
#EJH#         self.end_channel = 0  # Last channel to use
#EJH#         self.use_cursor = False  # Whether or not to use the cursor
#EJH#         self.determine_first_frames = (
#EJH#             False  # Are there blank frames before the shutter opens?
#EJH#         )
#EJH#         self.frame_avg_window = 1  # Number of frames to average over
#EJH#         self.sat_pixel_val = 10 ** 10  # Value representing saturated pixels
#EJH# 
#EJH#         self.task = ""
#EJH#         self.verbose = True
#EJH#         self.render_image = False
#EJH#         self.use_mask = False
#EJH#         self.seed_name = ""
#EJH# 
#EJH#         # Spots.find_in_frame
#EJH#         self.filter_image = "gaussian"
#EJH#         self.disk_radius = 5
#EJH#         self.bw_threshold_tolerance = 1.0
#EJH#         self.snr_filter_cutoff = 0.4
#EJH# 
#EJH#         self.max_displacement = 5.0
#EJH#         # Initialise
#EJH#         self.num_spots = 10
#EJH#         self.Isingle = 10000.0
#EJH#         self.BGmean = 500.0  # mean background pixel intensity
#EJH#         self.BGstd = 120.0  # standard deviation of background pixels
#EJH#         self.num_frames = 100
#EJH#         self.split_frame = False
#EJH#         self.frame_size = [64, 64]
#EJH# 
#EJH#         self.min_traj_len = 3
#EJH#         self.bleach_time = 0 # in frames, if 0 then no bleaching
#EJH#         self.diffusionCoeff = 1.0 # um2/s
#EJH# 
#EJH#         self.max_spot_molecules = 10
#EJH#         self.num_spot_molecules = None
#EJH#         self.nDiffPoints = 4  # number of MSD points to calculate diffusion const
#EJH#         self.frameTime = 0.005  # seconds
#EJH#         self.pixelSize = 0.120  # microns
#EJH#         self.PSFwidth = (
#EJH#             0.160 / self.pixelSize
#EJH#         )  # Sigma of a Gaussian, ~2/3 airy disk diameter
#EJH#         self.MSD_num_points = 4
#EJH# 
#EJH#         self.p_bleach_per_frame = 0.05
#EJH# 
#EJH#         self.subarray_halfwidth = 8
#EJH#         self.inner_mask_radius = 5
#EJH#         self.gauss_mask_sigma = 2.
#EJH#         self.gauss_mask_max_iter = 1000
#EJH# 
#EJH#         self.stoic_method = "linear_fit"
#EJH#         self.num_stoic_frames = 4
#EJH#         self.colocalize_n_frames = 5
#EJH#         
#EJH#         self.ALEX=False
#EJH#         self.start_channel='L'
#EJH#         self.colocalise=False
#EJH#         self.colocalise_distance = 5
#EJH#         self.overlap_thresh = 0.75
#EJH#         
#EJH#         self.calculate_isingle=True
#EJH#         self.copy_number=False

    def __getattr__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        else:
            try:
                return object.__getattribute__(self, "_params")[name]['value']
            except KeyError as exc:
                max_param = ''
                max_val  = 0
                for key in self._params:
                    if SequenceMatcher(None, name, key).ratio() > max_val:
                        max_param = key
                        max_val = SequenceMatcher(None, name, key).ratio()
                print(f"\nNo such key {name}. Did you mean {max_param}?\n")
                raise  exc

    def __setattribute__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._params[name]['value'] = value

    def help(self, name=None, param_class=None, level='basic'):
        names = []
        if name:
            names.append(name)
        elif param_class:
            for key in self._params:
                if level != 'advanced':
                    if (self._params[key]['class'] == param_class
                      and self._params[key]['level'] == 'basic'):
                        names.append(key)
                else:
                    names.append(key)
        elif level != 'basic':
            for key in self._params:
                names.append(key)
        else:
            for key in self._params:
                if self._params[key]['level'] == 'basic':
                    names.append(key)


        for name in names:
            print()
            print(f"{name.upper()}")
            print(f"  Description: {self._params[name]['description']}")
            print(f"      Default: {self._params[name]['default']}")
            print(f"        Class: {self._params[name]['class']}")
            print(f"        Level: {self._params[name]['level']}")



    def read(self, args):
        if "-r" in args:
            try:
                # Find where -r is and get the filename following it
                tag_index = args.index("-r")
                config_file = args.pop(tag_index + 1)
                args.pop(tag_index) # Remove the "-r" itself

                with open(config_file, 'r') as f:
                    # Parse the file: ignore comments, empty lines, and strip whitespace
                    file_args = []
                    for line in f:
                        line = line.split('#')[0].strip()
                        if line:
                            file_args.append(line)
                    
                args = file_args + args
            except (IndexError, FileNotFoundError) as e:
                sys.exit(f"ERROR: Could not read config file. {e}")

        # --- Standard Command Line Parsing ---
        if not args:
            return # Or sys.exit usage info

        self.task = args.pop(0).split(",")
        
        if self.task == ['help']:
            return
            
        if self.task != ['app'] and args:
            self.name = args.pop(0)

        # --- Key=Value Parsing for everything else ---
        for arg in args:
            if "=" not in arg:
                continue
                
            key, value = arg.split("=", 1)
            try:
                current_val = getattr(self, key)
                if isinstance(current_val, int):
                    setattr(self, key, int(value))
                elif isinstance(current_val, float):
                    setattr(self, key, float(value))
                elif isinstance(current_val, bool):
                    setattr(self, key, value == "True")
                elif isinstance(current_val, list):
                    setattr(self, key, list(map(lambda x: int(x), value.split(","))))
                else:
                    setattr(self, key, value)
            except (AttributeError, ValueError):
                print(f"Warning: Unknown or invalid parameter '{key}'")

            # Link pixel_size to psf_width automatically
            if key == "pixel_size":
                self.psf_width = 0.160 / self.pixel_size

    def param_dict(self, param_class=''):
        param_dict = {}

        
        if param_class:
            for k,v in self._params.items():
                if v["class"] == param_class:
                    param_dict[k] = v
        else:
            param_dict = self._params

        return param_dict


