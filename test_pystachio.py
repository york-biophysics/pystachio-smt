# -*- coding: utf-8 -*-
"""
PySTACHIO param testing script v0.2 JWS 2022
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys
import easygui

def write_params(pfile, params, values):
    f = open(pfile, "w")
    for i in range(len(params)):
        f.write(f'{params[i]:30}\t{values[i]}\n')
    f.close()

def make_param_list(params, values):
    p = ""
    for i in range(len(params)):
        p += params[i]+"="+str(values[i])+" "
    return p

def run_pystachio(task, f, p):
    if f[-4:] == '.tif':
        f = f[:-4]
    runstr = f'python.exe pystachio_smt.py {task} "{f}" {p}'
    print(runstr)
    try:
        os.system(runstr)
    except:
        easygui.exceptionbox(f'ERROR: something wrong running PySTACHIO on {f}\n')

def plotter(fname, pixel_data, start_frame, plot_frames, usechannel=None):
    if usechannel=='L':
        trajfile = fname[:-4]+"_Lchannel_trajectories.tsv"
    elif usechannel=='R':
        trajfile = fname[:-4]+"_Rchannel_trajectories.tsv"
    else:
        trajfile = fname[:-4]+"_trajectories.tsv"
    frame, x, y = np.loadtxt(trajfile, 
                              skiprows=1, 
                              usecols=(1,2,3), 
                              unpack=True)
    plot_frames = int(plot_frames)
    start_frame = int(start_frame)
    if plot_frames==-1 or plot_frames>pixel_data.shape[0]: 
        plot_frames=pixel_data.shape[0]
    for f in range(start_frame-1, start_frame+plot_frames-1):
        img = pixel_data[f,:,:].astype(np.uint16)    
        fig, ax = plt.subplots(1,2)
        if usechannel is not None:
            plt.title(f"{usechannel}-hand channel")
        ax[0].imshow(img, "Greys_r")
        ax[1].imshow(img, "Greys_r")
        ax[1].scatter(x[frame==f], y[frame==f], 100, 'r', marker='x')
        # ax[0].set_xlim([250,350])
        # ax[1].set_xlim([250,350])
        plt.show()

file = easygui.fileopenbox("Choose a file to test PySTCAHIO on").replace("\\", "/")

pfile = "parameters.txt"
params = ["num_frames", # 0
          "snr_filter_cutoff", #1
          "calculate_isingle", #2
          "I_single", #3
          "frame_time", #4
          "pixel_size", #5
          "min_traj_len", #6
          "struct_disk_radius", #7
          "msd_num_points", #8
          "max_displacement", #9
          "start_frame", #10
          "use_channel",
          "ALEX",
          "start_channel",
          "colocalize",
          "colocalize_n_frames",
          # "inner_mask_radius",
          # "subarray_halfwidth",
          "stoic_method",
          "num_stoic_frames",
          "copy_number",
          "mask_file",
          "num_procs",
          "chung_kennedy",
          # "psf_width",
          # "bw_threshold_tolerance",
          # "filter_image",
          # "gauss_mask_sigma",
          # "gauss_mask_iter"
          ]
defaults = [0,       # num_frames,
          0.4,        # "snr_filter_cutoff",
          "True",       # "calculate_isingle",
          10000,      # "I_single",
          0.005,      # "frame_time",
          0.12,       # "pixel_size",
          3,          # "min_traj_len",
          5,          # "struct_disk_radius",
          4,          # "msd_num_points",
          5.0,        # "max_displacement",
          0,          # "start_frame",
          "None",       # "use_channel",
          "False",      # "ALEX",
          "R",        # "start_channel",
          "False",      # "colocalize",
          5,          # "colocalize_n_frames",
          # 5,          # "inner_mask_radius",
          # 8,          # "subarray_halfwidth",
          'Linear',   # "stoic_method",
          3,          # "num_stoic_frames",
          "False",      # "copy_number",
          "None",       # "mask_file",
          1,          # "num_procs",
          "False" #chung_kennedy
          # 0.12,       # "psf_width",
          # 1,          # "bw_threshold_tolerance",
          # "Gaussian", # "filter_image",
          # 2,          # "gauus_mask_sigma",
          # 1000,       # "gauss_mask_iter"
          ]
if os.path.exists(pfile):
    tmp = np.loadtxt(pfile, unpack=True, usecols=(1,), dtype=str)
    if len(tmp) == len(defaults):
        defaults = tmp
values = easygui.multenterbox("Enter tracking/postprocessing parameters", "Input params", params, defaults)
# # Build parameters list from text box input
write_params(pfile, params, values)
p = make_param_list(params, values)

os.chdir("pystachio_smt")
run_pystachio('track,postprocess', file, p)




pixel_data = tifffile.imread(file)
plot_frames = int(values[0]) #0/-1 for all
start_frame = values[10]
channel=values[11]
if channel=="L":
    pixel_data = pixel_data[:,:,:pixel_data.shape[2]//2]
    try:
        plotter(file, pixel_data, start_frame, values[0])
    except:
        print("WARNING: No trajectories found")
elif channel=="R":
    pixel_data = pixel_data[:,:,pixel_data.shape[2]//2:]
    try:
        plotter(file, pixel_data, start_frame, values[0])
    except:
        print("WARNING: No trajectories found")
elif values[11]=="None" and values[12]=="True":
    if values[13]=='R':
        Ldata = pixel_data[1:11:2,:,:]
        Rdata = pixel_data[0:10:2,:,:]
    else:
        Ldata = pixel_data[0:10:2,:,:]
        Rdata = pixel_data[1:11:2,:,:]
    try:
        plotter(file, Ldata, start_frame, values[0], usechannel='L')
        # frame, x, y = np.loadtxt(file[:-4]+"_Lchannel_trajectories.tsv", 
        #                           skiprows=1, 
        #                           usecols=(1,2,3), 
        #                           unpack=True)
        # if plot_frames==0 or plot_frames>pixel_data.shape[0]: 
        #     plot_frames=pixel_data.shape[0]
        # for f in range(plot_frames//2):
        #     img = Ldata[f,:,:].astype(np.uint16)    
        #     fig, ax = plt.subplots(1,2)
        #     ax[0].imshow(img, "Greys_r")
        #     ax[1].imshow(img, "Greys_r")
        #     ax[1].scatter(x[frame==f], y[frame==f], 100, 'r', marker='x')
        #     plt.title(f"Left hand channel frame {f}")
        #     # ax[0].set_xlim([250,350])
        #     # ax[1].set_xlim([250,350])
        #     plt.show()
    except:
        print("WARNING: No left hand channel trajectories found!")
    try:
        plotter(file, Rdata, start_frame, values[0], usechannel='R')
        # frame, x, y = np.loadtxt(file[:-4]+"_Rchannel_trajectories.tsv", 
        #                           skiprows=1, 
        #                           usecols=(1,2,3), 
        #                           unpack=True)
        # if plot_frames==-1 or plot_frames>pixel_data.shape[0]: 
        #     plot_frames=pixel_data.shape[0]
        # for f in range(plot_frames//2):
        #     img = Rdata[f,:,:].astype(np.uint16)    
        #     fig, ax = plt.subplots(1,2)
        #     ax[0].imshow(img, "Greys_r")
        #     ax[1].imshow(img, "Greys_r")
        #     ax[1].scatter(x[frame==f]+Rdata.shape[2]//2, y[frame==f], 100, 'r', marker='x')
        #     plt.title(f"Right hand channel frame {f}")
        #     # ax[0].set_xlim([250,350])
        #     # ax[1].set_xlim([250,350])
        #     plt.show()
    except:
        print("WARNING: No right hand channel trajectories found!")
    
else:
    try:
        plotter(file, pixel_data, start_frame, values[0])
    except:
        print("WARNING: No trajectories found")
