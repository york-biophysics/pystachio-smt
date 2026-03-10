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
import easygui

def refresh(event):
    global coords
    global trajs
    global params
    global do_stoic
    global active_frame
    global all_spots
    global frame_box
    ax2.set_prop_cycle(None)
    ax2.clear()

    num_frames = int(frames_box.text)
    params.num_frames = int(frames_box.text)
    if ck_radio.value_selected == 'Yes':
        params.chung_kennedy=True
    else:
        params.chung_kennedy=False
    if stoic_radio.value_selected == 'Yes':
        do_stoic=True
    else:
        do_stoic=False
    params.Isingle = int(isingle_box.text)

    ax1.clear()
    ax1.imshow(image_data[active_frame].as_image(), cmap='Greys_r')
    for coord in all_spots[0].positions:
        ax1.scatter(coord[0], coord[1], marker='x')
    for traj in trajs:
        t = np.array(traj.intensity)
        label=''
        if do_stoic:
            i = np.mean(traj.intensity[:3])
            stoici = i/params.Isingle
            label = f'Estimated stoic: {stoici:.1f}'
        if params.chung_kennedy:
            ck_data = postprocessing.chung_kennedy_filter(t,params.chung_kennedy_window,1)[0][:-1]
            ax2.plot(ck_data[:num_frames-1]/10**3, label=label)
            ax2.set_title("Chung-Kennedy filtered intensity data")
        else:
            ax2.set_title("Raw intensity data")
            ax2.plot(t[:num_frames]/10**3, label=label)
    if do_stoic:
        ax2.legend()
    ax2.set_xlabel("Frame number")
    ax2.set_ylabel("Intensity (camera counts per pixel x$10^3$)")
    fig.canvas.draw_idle()
    
def update(val):
    global active_frame
    global coords
    active_frame = int(frame_slider.val)
    ax1.clear()
    ax1.imshow(image_data[active_frame].as_image(), cmap='Greys_r')
    for coord in all_spots[0].positions:
        ax1.scatter(coord[0], coord[1], marker='x')
    fig.canvas.draw_idle()

def onclick(event):
    # Ignore clicks outside axes
    if event.inaxes != ax1 or event.xdata is None or event.ydata is None:
        return

    toolbar = getattr(fig.canvas, "toolbar", None)
    if toolbar is not None and getattr(toolbar, "mode", ""):
        return

    global ix, iy
    global coords
    global trajs
    global params
    global do_stoic
    global active_frame
    global all_spots
    global frames_box
    ax2.set_prop_cycle(None)
    ax2.clear()

    frames_to_show = int(frames_box.text)
    params.num_frames = int(image_data.num_frames)
       
    if ck_radio.value_selected == 'Yes':
        params.chung_kennedy=True
    else:
        params.chung_kennedy=False
    if stoic_radio.value_selected == 'Yes':
        do_stoic=True
    else:
        do_stoic=False
    params.Isingle = int(isingle_box.text)

    ax1.clear()
    ax1.imshow(image_data[active_frame].as_image(), cmap='Greys_r')
    ix, iy = event.xdata, event.ydata
#     coords.append((ix, iy))
    # for coord in coords:
    #     ax1.scatter(coord[0], coord[1], marker='x')

    tmp_spots = spots.Spots(frame=active_frame)
    frame_data = image_data[active_frame]
    # frame_spots.find_in_frame(frame_data.as_image()[:, :], params)
    tmp_spots.set_positions([(ix,iy)])
    tmp_spots.num_spots = 1
    tmp_spots.refine_centres(frame_data, params)
    coords.append(tmp_spots.positions[:])
    
    all_spots = []
    clicked_spots = len(coords)
    for frame in range(params.num_frames):
        frame_spots = spots.Spots(frame=frame)
        frame_data = image_data[frame]
        # frame_spots.find_in_frame(frame_data.as_image()[:, :], params)
        frame_spots.set_positions(coords)
        frame_spots.num_spots = clicked_spots
        frame_spots.get_spot_intensities(frame_data.as_image()[:,:], params)
        all_spots.append(frame_spots)
    for coord in all_spots[0].positions:
        ax1.scatter(coord[0], coord[1], marker='x')

    trajs = trajectories.build_trajectories(all_spots, params)

    for traj in trajs:
        t = np.array(traj.intensity)
        label=''
        if do_stoic:
            i = np.mean(traj.intensity[:3])
            stoici = i/params.Isingle
            label = f'Estimated stoic: {stoici:.1f}'
        if params.chung_kennedy:
            ck_data = postprocessing.chung_kennedy_filter(t,params.chung_kennedy_window,1)[0][:-1]
            ax2.plot(ck_data[:frames_to_show-1]/10**3, label=label)
            ax2.set_title("Chung-Kennedy filtered intensity data")
        else:
            ax2.set_title("Raw intensity data")
            ax2.plot(t[:frames_to_show]/10**3, label=label)
    if do_stoic:
        ax2.legend()
    ax2.set_xlabel("Frame number")
    ax2.set_ylabel("Intensity (camera counts per pixel x$10^3$)")
    fig.canvas.draw_idle()

def pick_file(event):
    fname = easygui.fileopenbox()
    global params
    global image_data
    global coords
    params = parameters.Parameters()
    image_data = images.ImageData()
    params.name = fname[:-4]
    image_data.read(fname, params)
    ax1.clear()
    ax2.clear()
    coords = []
    ax1.imshow(image_data[0].as_image(), cmap='Greys_r')
    fig.canvas.draw_idle()
    return 0

def run_gui(event):
    import subprocess
    import sys
    if 'fname' in locals() or 'fname' in globals():
        subprocess.Popen(['python3', 'gui.py',f'{fname}'])
    else:
        subprocess.Popen(['python3', 'gui.py'])        
    sys.exit()

def save(event):
    rootname = params.name.rsplit('/',1)[0]
    # print(params.name)
    outname = rootname + '/' + outname_box.text
    trajectories.write_trajectories(trajs, outname+".tsv")
    # print(trajs, outname)
    tmpfig,tmpax = plt.subplots()
    for traj in trajs:
        t = np.array(traj.intensity)
        label=''
        if do_stoic:
            i = np.mean(traj.intensity[:3])
            stoici = i/params.Isingle
            label = f'Estimated stoic: {stoici:.1f}'
        if params.chung_kennedy:
            ck_data = postprocessing.chung_kennedy_filter(t,params.chung_kennedy_window,1)[0][:-1]
            tmpax.set_title("Chung-Kennedy filtered intensity data")
            tmpax.plot(ck_data/10**3, label=label)
        else:
            tmpax.plot(t/10**3, label=label)
            tmpax.set_title("Raw intensity data")
    if do_stoic:
        tmpax.legend()
    tmpax.set_xlabel("Frame number")
    tmpax.set_ylabel("Intensity (camera counts per pixel x$10^3$)")
    tmpfig.savefig(outname+'.png', dpi=300)
    plt.close(tmpfig)
    return 0

def clear_plot(event):
    global coords
    coords = []
    ax1.clear()
    ax1.imshow(image_data[0].as_image(), cmap='Greys_r')
    ax2.clear()
    fig.canvas.draw_idle()



fig,(ax1,ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [1,2]})
fig.subplots_adjust(right=0.7, bottom=0.25)

active_frame = 0
if len(sys.argv) == 2:
    fname = sys.argv[1]
    params = parameters.Parameters()
    image_data = images.ImageData()
    params.name = fname[:-4]
    image_data.read(fname, params)
    ax1.clear()
    ax2.clear()
    coords = []
    ax1.imshow(image_data[0].as_image(), cmap='Greys_r')
    fig.canvas.draw_idle()
else:
    fname = easygui.fileopenbox()
    params = parameters.Parameters()
    image_data = images.ImageData()
    params.name = fname[:-4]
    image_data.read(fname, params)
    ax1.clear()
    ax2.clear()
    coords = []
    ax1.imshow(image_data[0].as_image(), cmap='Greys_r')
    fig.canvas.draw_idle()

frames_ax = plt.axes([0.75, 0.85, 0.15, 0.075])
frames_ax.set_title("Number of frames")
frames_box = TextBox(frames_ax, '', initial=image_data.num_frames)
slider_ax = fig.add_axes([0.2, 0.1, 0.5, 0.03])
frame_slider = Slider(
    ax=slider_ax,
    label='Frame',
    valmin=0,
    valmax=image_data.num_frames,
    valinit=0,
)
frame_slider.on_changed(update)

ck_ax = plt.axes([0.75, 0.7, 0.15, 0.075])
ck_ax.set_title("Chung-Kennedy?")
ck_radio = RadioButtons(ck_ax, ('Yes', 'No'), active=0)

stoic_ax = plt.axes([0.75, 0.55, 0.15, 0.075])
stoic_ax.set_title("Estimate stoichiometry??")
stoic_radio = RadioButtons(stoic_ax, ('Yes', 'No'), active=1)

isingle_ax = plt.axes([0.75, 0.4, 0.15, 0.075])
isingle_ax.set_title("Estimate Isingle")
isingle_box = TextBox(isingle_ax, '', initial="120")

file_ax = plt.axes([0.75, 0.25, 0.05, 0.075])
file_button = Button(file_ax, "Pick file")
clear_ax = plt.axes([0.8, 0.25, 0.05, 0.075])
clear_button = Button(clear_ax, "Clear")
refresh_ax = plt.axes([0.85, 0.25, 0.05, 0.075])
refresh_button = Button(refresh_ax, "Refresh")
gui_ax = plt.axes([0.9, 0.25, 0.05, 0.075])
gui_button = Button(gui_ax, "Run full GUI")

outname_ax = plt.axes([0.75, 0.1, 0.1, 0.075])
outname_ax.set_title("Output name")
outname_box = TextBox(outname_ax, '', initial='clickmode_output') 
save_ax = plt.axes([0.85, 0.1, 0.075, 0.075])
save_button = Button(save_ax, "Save")

file_button.on_clicked(pick_file)
save_button.on_clicked(save)
clear_button.on_clicked(clear_plot)
gui_button.on_clicked(run_gui)
refresh_button.on_clicked(refresh)
    
coords = []
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
fig.canvas.mpl_disconnect(cid)



