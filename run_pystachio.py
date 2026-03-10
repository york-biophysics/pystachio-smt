import numpy as np
import easygui
import glob
import os

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

wdir = easygui.diropenbox("Select folder to work in").replace("\\","/")

taskopts = ("Both", "Track", "Postprocess")
task = easygui.choicebox("Select your tasks(s)", "Select task", taskopts)
if task==taskopts[0]:
    task="track,postprocess "
elif task==taskopts[1]:
    task="track "
elif task==taskopts[2]:
    task="postprocess "

options = ("Analyse all subfolders in this folder", "Analyse EVEN numbered folders (assumes _2 etc)", "Analyse ODD numbered folder (assumes _1 etc)", "Analyse all files in this folder", "Analyse all subfolders (merged data)")
option = easygui.choicebox("What do you want to analyse?", "Choose option", options)
if option==options[0]:
    option='all'
    merged = False
elif option==options[4]:
    option='all'
    merged=True
elif option==options[1]:
    option='even'
elif option==options[2]:
    option='odd'
elif option==options[3]:
    option='files'

# Build p here with multenterbox
pfile = "parameters.txt"
params = ["num_frames",
          "snr_filter_cutoff",
          "calculate_isingle",
          "I_single",
          "frame_time",
          "pixel_size",
          "min_traj_len",
          "struct_disk_radius",
          "msd_num_points",
          "max_displacement",
          "start_frame",
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
          "chung_kennedy"
          # "psf_width",
          # "bw_threshold_tolerance",
          # "filter_image",
          # "gauus_mask_sigma",
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
          1 ,         # "num_procs",
          "False"
          # 0.12,       # "psf_width",
          # 1,          # "bw_threshold_tolerance",
          # "Gaussian", # "filter_image",
          # 2,          # "gauus_mask_sigma",
          # 1000,       # "gauss_mask_iter"
          ]

if os.path.exists(pfile):
    try:
        tmp = np.loadtxt(pfile, unpack=True, usecols=(1,), dtype=str)
        if len(tmp)==len(defaults):
            defaults = tmp
    except:
        print("Dodgy parameters file, using defaults")
values = easygui.multenterbox("Enter tracking/postprocessing parameters", "Input params", params, defaults)
write_params(pfile, params, values)
p = make_param_list(params, values)

os.chdir("pystachio_smt")
if option=='files':
    files = glob.glob(wdir+"/*tif")
    for f in files:
        run_pystachio(task, f, p)
elif option=='all':
    dirs = glob.glob(wdir+"/*/")
    for d in dirs:
        files = glob.glob(d+"*tif")
        for f in files:
            if merged:
                n = f.split("_")[-3]
                if not any(c.isalpha() for c in n):
                    n = int(n)
                    if n%2==0:
                        run_pystachio(task, f, p)
            else:
                run_pystachio(task, f, p)
else:
    dirs = glob.glob(wdir+"/*/")
    for d in dirs:
        try:
            oddeven = int(d.replace("\\","/").split("_")[-1][:-1])%2
            if oddeven==1 and option=='odd':
                f = glob.glob(d+"*tif")[0][:-4]
                run_pystachio(task, f, p)
            elif oddeven==0 and option=='even':
                f = glob.glob(d+"*tif")[0][:-4]
                run_pystachio(task, f, p)
        except:
            print(f"WARNING: messed up on directory {d} - maybe didn't have a number at the end?")
write_params(wdir+"/pystachio_settings.txt", params, values)
