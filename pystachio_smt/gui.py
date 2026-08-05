import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, RadioButtons, Slider
import numpy as np
import tifffile as tf
import images
import spots
import parameters
import postprocessing
import trajectories
import tracking
import easygui

def run_clickmode(event):
    global fname
    import subprocess
    import sys
    if 'fname' in locals() or 'fname' in globals():
        subprocess.Popen(['python3','clickmode.py', f'{fname}'])
    else:
        subprocess.Popen(['python3','clickmode.py'])        
    sys.exit()

def update(val):
    frame = int(frame_slider.val)
    ax.clear()
    ax.imshow(image_data[frame].as_image(), cmap='Greys_r')
    fig.canvas.draw_idle()

def pick_file(event):
    global fname
    global params
    global image_data
    fname = easygui.fileopenbox()
    params = parameters.Parameters()
    image_data = images.ImageData()
    params.name = fname[:-4]
    image_data.read(fname, params)
    ax.clear()
    ax.imshow(image_data[0].as_image(), cmap='Greys_r')
    frame_slider.valmax = image_data.num_frames
    fig.canvas.draw_idle()

def find_spots(event):
    global fname
    global params
    global image_data
    ax.clear()
    params.frame_time = float(frame_time_box.text)
    params.pixel_size = float(pixel_size_box.text)
    params.start_frame = int(start_frame_box.text)
    if calculate_isingle_radio.value_selected == 'Yes':
        params.calculate_isingle = True
    else:
        params.calculate_isingle = False
    params.I_single = float(isingle_box.text)
    params.snr_filter_cutoff = float(snr_box.text)
    params.max_displacement = int(maxdisp_box.text)
    params.min_traj_len = int(mintraj_box.text)
    params.bw_threshold_tolerance = float(bwthresh_box.text)
    params.struct_disk_radius = int(sdisk_box.text)
    params.subarray_halfwidth = int(subarray_box.text)
    params.inner_mask_radius = int(innermask_box.text)
    frame = int(frame_slider.val)
    ax.imshow(image_data[frame].as_image(), cmap='Greys_r')
    all_spots = tracking.track_frame(image_data[frame], frame, params)
    pos = all_spots.positions
    for i in range(len(pos)):
        ax.scatter(pos[i][0],pos[i][1],marker='x')
    fig.canvas.draw_idle()

def run_pystachio(event):
    global fname
    global params
    global image_data
    if show_figs_radio.value_selected == 'Yes':
        params.display_figures = True
    else:
        params.display_figures = False
    params.num_procs = int(num_procs_box.text)
    params.frame_time = float(frame_time_box.text)
    params.pixel_size = float(pixel_size_box.text)
    params.start_frame = int(start_frame_box.text)
    if calculate_isingle_radio.value_selected == 'Yes':
        params.calculate_isingle = True
    else:
        params.calculate_isingle = False
    params.I_single = float(isingle_box.text)
    params.snr_filter_cutoff = float(snr_box.text)
    params.max_displacement = int(maxdisp_box.text)
    params.min_traj_len = int(mintraj_box.text)
    params.bw_threshold_tolerance = float(bwthresh_box.text)
    params.struct_disk_radius = int(sdisk_box.text)
    params.subarray_halfwidth = int(subarray_box.text)
    params.inner_mask_radius = int(innermask_box.text)
    if ALEX_radio.value_selected == 'Yes':
        params.ALEX = True
    else:
        params.ALEX = False
    plt.close()
    global run
    run = True

def get_ALEX_params(event):
    global params
    msg = 'Enter ALEX tracking parameters'
    title = 'ALEX parameters'
    input_list = ['start_channel (L/R)', 'Lisingle', 'Risingle', 'colocalize (True/False)', 'colocalize_distance', 'colocalize_n_frames']
    defaults = [params.start_channel,params.L_isingle,params.R_isingle,params.colocalize,params.colocalize_distance,params.colocalize_n_frames]
    vals = easygui.multenterbox(msg, title, input_list, defaults)
#    print(vals)
    params.start_channel = vals[0]
    params.L_isingle = float(vals[1])
    params.R_isingle = float(vals[2])
    params.colocalize = vals[3]
    params.colocalize_distance = int(vals[4])
    params.colocalize_n_frames = int(vals[5])
    
def defaults(event):
    num_procs_box.set_val('1')    
    frame_time_box.set_val('0.005')
    pixel_size_box.set_val('0.120')
    start_frame_box.set_val('0')
    calculate_isingle_radio.set_active(1)
    show_figs_radio.set_active(0)
    isingle_box.set_val('120')
    snr_box.set_val('0.4')
    maxdisp_box.set_val('5')
    mintraj_box.set_val('3')
    bwthresh_box.set_val('1.0')
    sdisk_box.set_val('5')
    subarray_box.set_val('8')
    innermask_box.set_val('5')
    ALEX_radio.set_active(1)
    params.start_channel = 'L'
    params.L_isingle = 10000
    params.R_isingle = 10000
    params.colocalise = False
    params.colocalize_distance = 5
    params.colocalise_n_frames = 5
    fig.canvas.draw_idle()
    
fig,ax = plt.subplots()
fig.subplots_adjust(right=0.7, bottom=0.25)

slider_ax = fig.add_axes([0.2, 0.1, 0.5, 0.03])
frame_slider = Slider(
    ax=slider_ax,
    label='Frame',
    valmin=0,
    valmax=1000,
    valinit=0,
)

num_procs_ax = plt.axes([0.75, 0.9, 0.05, 0.05])
num_procs_box = TextBox(num_procs_ax, 'num_procs', initial="1")
frame_time_ax = plt.axes([0.75, 0.85, 0.05, 0.05])
frame_time_box = TextBox(frame_time_ax, 'frame_time', initial="0.005")
pixel_size_ax = plt.axes([0.75, 0.8, 0.05, 0.05])
pixel_size_box = TextBox(pixel_size_ax, 'pixel_size', initial="0.120")
start_frame_ax = plt.axes([0.75, 0.75, 0.05, 0.05])
start_frame_box = TextBox(start_frame_ax, 'start_frame', initial="0")
calculate_isingle_ax = plt.axes([0.75, 0.65, 0.05, 0.05])
calculate_isingle_ax.set_title("Calculate Isingle?")
calculate_isingle_radio = RadioButtons(calculate_isingle_ax, ('Yes', 'No'), active=1)
isingle_ax = plt.axes([0.75, 0.6, 0.05, 0.05])
isingle_box = TextBox(isingle_ax, 'I_single', initial="120")
snr_ax = plt.axes([0.75, 0.5, 0.05, 0.05])
snr_box = TextBox(snr_ax, 'SNR cutoff', initial="0.4")
ALEX_ax = plt.axes([0.75, 0.4, 0.05, 0.05])
ALEX_ax.set_title("ALEX?")
ALEX_radio = RadioButtons(ALEX_ax, ('Yes', 'No'), active=1)

maxdisp_ax = plt.axes([0.9, 0.85, 0.05, 0.05])
maxdisp_box = TextBox(maxdisp_ax, 'max_displacement', initial="5")
mintraj_ax = plt.axes([0.9, 0.8, 0.05, 0.05])
mintraj_box = TextBox(mintraj_ax, 'min_traj_length', initial="3")
bwthresh_ax = plt.axes([0.9, 0.75, 0.05, 0.05])
bwthresh_box = TextBox(bwthresh_ax, 'bw_threshold_tolerance', initial="1.0")
show_figs_ax = plt.axes([0.9, 0.65, 0.05, 0.05])
show_figs_ax.set_title("Show figures?")
show_figs_radio = RadioButtons(show_figs_ax, ('Yes', 'No'), active=0)
sdisk_ax = plt.axes([0.9, 0.55, 0.05, 0.05])
sdisk_box = TextBox(sdisk_ax, 'struct_disk_radius', initial="5")
subarray_ax = plt.axes([0.9, 0.5, 0.05, 0.05])
subarray_box = TextBox(subarray_ax, 'subarray_halfwidth', initial="8")
innermask_ax = plt.axes([0.9, 0.45, 0.05, 0.05])
innermask_box = TextBox(innermask_ax, 'inner_mask_radius', initial="5")

file_ax = plt.axes([0.75, 0.3, 0.075, 0.075])
file_button = Button(file_ax, "Pick file")
clickmode_ax = plt.axes([0.875, 0.3, 0.075, 0.075])
clickmode_button = Button(clickmode_ax, "Click mode")

ALEX_ax = plt.axes([0.75, 0.2, 0.075, 0.075])
ALEX_button = Button(ALEX_ax, "ALEX")
defaults_ax = plt.axes([0.875, 0.2, 0.075, 0.075])
defaults_button = Button(defaults_ax, "Defaults")

spots_ax = plt.axes([0.75, 0.1, 0.075, 0.075])
spots_button = Button(spots_ax, "Find spots")
analyse_ax = plt.axes([0.875, 0.1, 0.075, 0.075])
analyse_button = Button(analyse_ax, "Run full analysis")

file_button.on_clicked(pick_file)
spots_button.on_clicked(find_spots)
defaults_button.on_clicked(defaults)
analyse_button.on_clicked(run_pystachio)
ALEX_button.on_clicked(get_ALEX_params)
clickmode_button.on_clicked(run_clickmode)
frame_slider.on_changed(update)

if len(sys.argv) == 2:
    fname = sys.argv[1]
    params = parameters.Parameters()
    image_data = images.ImageData()
    params.name = fname[:-4]
    image_data.read(fname, params)
    ax.clear()
    ax.imshow(image_data[0].as_image(), cmap='Greys_r')
    frame_slider.valmax = image_data.num_frames
    fig.canvas.draw_idle()
else:
    fname = easygui.fileopenbox()
    params = parameters.Parameters()
    image_data = images.ImageData()
    params.name = fname[:-4]
    image_data.read(fname, params)
    ax.clear()
    ax.imshow(image_data[0].as_image(), cmap='Greys_r')
    frame_slider.valmax = image_data.num_frames
    fig.canvas.draw_idle()
plt.show()

if 'run' in locals() or 'run' in globals():
    if run:
        tracking.track(params)
        postprocessing.postprocess(params)
