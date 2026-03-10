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

def get_data(fname):
    global donor_data
    global acceptor_data
    global fret_data
    global coords
    coords = []

    donor_data = images.ImageData()
    donor_data.read(fname, params)
    acceptor_data = images.ImageData()
    acceptor_data.read(fname, params)
    fret_data = images.ImageData()
    fret_data.read(fname, params)
    donor=np.zeros((donor_data.num_frames//2,donor_data.frame_size[1],donor_data.frame_size[0]//2))
    acc=np.zeros((donor_data.num_frames//2,donor_data.frame_size[1],donor_data.frame_size[0]//2))
    fret=np.zeros((donor_data.num_frames//2,donor_data.frame_size[1],donor_data.frame_size[0]//2))
    for i in range(0,donor_data.num_frames-1,2):
        donor[i//2,:,:] = donor_data.pixel_data[i,:,:donor_data.frame_size[0]//2]
        fret[i//2,:,:] = donor_data.pixel_data[i,:,donor_data.frame_size[0]//2:]
        acc[i//2,:,:] = acceptor_data.pixel_data[i+1,:,acceptor_data.frame_size[0]//2:]
    donor_data.num_frames = donor_data.num_frames//2
    donor_data.pixel_data = donor
    donor_data.frame_size = [donor_data.frame_size[0]//2,donor_data.frame_size[1]]
    fret_data.num_frames = fret_data.num_frames//2
    fret_data.pixel_data = fret
    fret_data.frame_size = [fret_data.frame_size[0]//2,fret_data.frame_size[1]]
    acceptor_data.num_frames = donor_data.num_frames//2
    acceptor_data.pixel_data = acc
    acceptor_data.frame_size = [acceptor_data.frame_size[0]//2,acceptor_data.frame_size[1]]

def draw_images(frame=0):
    global cb1
    global cb2
    global cb3
    if frame >= donor_data.num_frames: return 0
    donor_im.clear()
    acc_im.clear()
    fret_im.clear()
    if 'cb1' in locals() or 'cb1' in globals():
        cb1.remove()
        cb2.remove()
        cb3.remove()
    im1 = donor_im.imshow(donor_data[frame].as_image(), cmap='Greys_r')
    donor_im.set_xticks([])
    donor_im.set_yticks([])
    im2 = acc_im.imshow(acceptor_data[frame].as_image(), cmap='Greys_r')
    acc_im.set_xticks([])
    acc_im.set_yticks([])
    im3 = fret_im.imshow(fret_data[frame].as_image(), cmap='Greys_r')
    fret_im.set_xticks([])
    fret_im.set_yticks([])
    iplot_donor.clear()
    donor_im.set_title("Donor channel")
    acc_im.set_title("Acceptor channel")
    fret_im.set_title("FRET channel")
    cb1 = fig.colorbar(im1, ax=(donor_im))
    cb2 = fig.colorbar(im2, ax=(acc_im))
    cb3 = fig.colorbar(im3, ax=(fret_im))
    fig.canvas.draw_idle()

def scatter_coords():
    global coords
    for coord in coords:
        donor_im.scatter(coord[0], coord[1], marker='x')
        acc_im.scatter(coord[0], coord[1], marker='x')
        fret_im.scatter(coord[0], coord[1], marker='x')
    fig.canvas.draw_idle()

def plot_trajectory_intensities():
    global trajs_donor
    global trajs_acc
    global trajs_fret
    global iplot_donor
    global iplot_acc
    global iplot_fret
    
    global frame_box
    num_frames = int(frames_box.text)

    for traj in range(len(trajs_donor)):
        tdonor = np.array(trajs_donor[traj].intensity)
        tacc = np.array(trajs_acceptor[traj].intensity)
        tfret = np.array(trajs_fret[traj].intensity)
        label_donor=''
        label_acc=''
        label_fret=''
        if params.chung_kennedy:
            ck_donor = postprocessing.chung_kennedy_filter(tdonor,params.chung_kennedy_window,1)[0][:-1]
            ck_acc = postprocessing.chung_kennedy_filter(tacc,params.chung_kennedy_window,1)[0][:-1]
            ck_fret = postprocessing.chung_kennedy_filter(tfret,params.chung_kennedy_window,1)[0][:-1]
            iplot_donor.plot(ck_donor[:num_frames-1]/10**3, label=label_donor)
            iplot_acc.plot(ck_acc[:num_frames-1]/10**3, label=label_acc)
            iplot_fret.plot(ck_fret[:num_frames-1]/10**3, label=label_acc)
            plot_efret.plot(ck_fret[:num_frames-1]/ck_donor[:num_frames-1])
        else:
            iplot_donor.plot(tdonor[:num_frames]/10**3, label=label_donor)
            iplot_acc.plot(tacc[:num_frames]/10**3, label=label_acc)
            iplot_fret.plot(tfret[:num_frames]/10**3, label=label_fret)
            plot_efret.plot(tfret[:num_frames]/tdonor[:num_frames])
    plot_efret.set_xlabel("Frame number")
    iplot_acc.set_ylabel(r"Intensity (x10$^3$)")
    fig.canvas.draw_idle()
    
def refresh(event):
    global coords
    global trajs_donor
    global trajs_acceptor
    global trajs_fret
    global params
    global do_stoic
    global active_frame
    global frame_box
    iplot_donor.set_prop_cycle(None)
    iplot_donor.clear()
    iplot_acc.set_prop_cycle(None)
    iplot_acc.clear()
    iplot_fret.set_prop_cycle(None)
    iplot_fret.clear()
    plot_efret.set_prop_cycle(None)
    plot_efret.clear()

    params.num_frames = int(frames_box.text)        
    if ck_radio.value_selected == 'Yes':
        params.chung_kennedy=True
    else:
        params.chung_kennedy=False

    draw_images()
    scatter_coords()
    plot_trajectory_intensities()

def update(val):
    global active_frame
    global coords
    active_frame = int(frame_slider.val)
    draw_images(active_frame)
    scatter_coords()

def onclick(event):
    # Ignore clicks outside axes
    if event.inaxes != donor_im or event.xdata is None or event.ydata is None:
        if event.inaxes!= acc_im:
            if event.inaxes!=fret_im: 
                return

    toolbar = getattr(fig.canvas, "toolbar", None)
    if toolbar is not None and getattr(toolbar, "mode", ""):
        return

    global ix, iy
    global coords
    global trajs_donor
    global trajs_acceptor
    global trajs_fret
    global params
    global do_stoic
    global active_frame
    iplot_donor.set_prop_cycle(None)
    iplot_donor.clear()
    iplot_acc.set_prop_cycle(None)
    iplot_acc.clear()
    iplot_fret.set_prop_cycle(None)
    iplot_fret.clear()

    params.num_frames = int(frames_box.text)
    if params.num_frames + active_frame > donor_data.num_frames:
        end_frame = donor_data.num_frames
    else:
        end_frame = params.num_frames + active_frame
        
    if ck_radio.value_selected == 'Yes':
        params.chung_kennedy=True
    else:
        params.chung_kennedy=False

    draw_images(active_frame)
    
    donor_spots = []
    acc_spots = []
    fret_spots = []
    ix, iy = event.xdata, event.ydata
    # Refine clicked centres
    if event.inaxes == donor_im:
        tmp_spots = spots.Spots(frame=active_frame)
        frame_data = donor_data[active_frame]
        tmp_spots.set_positions([(ix,iy)])
        tmp_spots.num_spots = 1
        tmp_spots.refine_centres(frame_data, params)
        coords.append(tmp_spots.positions[0][:])
    elif event.inaxes == acc_im:
        tmp_spots = spots.Spots(frame=active_frame)
        frame_data = acceptor_data[active_frame]
        tmp_spots.set_positions([(ix,iy)])
        tmp_spots.num_spots = 1
        tmp_spots.refine_centres(frame_data, params)
        coords.append(tmp_spots.positions[0][:])
    else:
        tmp_spots = spots.Spots(frame=active_frame)
        frame_data = fret_data[active_frame]
        tmp_spots.set_positions([(ix,iy)])
        tmp_spots.num_spots = 1
        tmp_spots.refine_centres(frame_data, params)
        coords.append(tmp_spots.positions[0][:])

    clicked_spots = len(coords)
    scatter_coords()
    
    for frame in range(params.num_frames):
        frame_spots_donor = spots.Spots(frame=frame)
        frame_spots_acc = spots.Spots(frame=frame)
        frame_spots_fret = spots.Spots(frame=frame)
        
        frame_data_donor = donor_data[frame]
        frame_data_acc = acceptor_data[frame]
        frame_data_fret = fret_data[frame]

        frame_spots_donor.set_positions(coords)
        frame_spots_acc.set_positions(coords)
        frame_spots_fret.set_positions(coords)

        frame_spots_donor.num_spots = clicked_spots
        frame_spots_acc.num_spots = clicked_spots
        frame_spots_fret.num_spots = clicked_spots

        frame_spots_donor.get_spot_intensities(frame_data_donor.as_image()[:,:], params)
        frame_spots_acc.get_spot_intensities(frame_data_acc.as_image()[:,:], params)
        frame_spots_fret.get_spot_intensities(frame_data_fret.as_image()[:,:], params)

        donor_spots.append(frame_spots_donor)
        acc_spots.append(frame_spots_acc)
        fret_spots.append(frame_spots_fret)
    trajs_donor = trajectories.build_trajectories(donor_spots, params)
    trajs_acceptor = trajectories.build_trajectories(acc_spots, params)
    trajs_fret = trajectories.build_trajectories(fret_spots, params)

    plot_trajectory_intensities()

def pick_file(event):
    fname = easygui.fileopenbox()
    params = parameters.Parameters()
    params.name = fname[:-4]
    get_data(fname)
    draw_images()

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
    outname = rootname + '/' + outname_box.text
    trajectories.write_trajectories(trajs_donor, outname+".tsv")
    tmpfig,tmpax = plt.subplots()
    for traj in trajs_donor:
        t = np.array(traj.intensity)
        label=''
        if params.chung_kennedy:
            ck_data = postprocessing.chung_kennedy_filter(t,params.chung_kennedy_window,1)[0][:-1]
            tmpax.set_title("Chung-Kennedy filtered intensity data")
            tmpax.plot(ck_data/10**3, label=label)
        else:
            tmpax.plot(t/10**3, label=label)
            tmpax.set_title("Raw intensity data")
    tmpax.set_xlabel("Frame number")
    tmpax.set_ylabel("Intensity (camera counts per pixel x$10^3$)")
    tmpfig.savefig(outname+'.png', dpi=300)
    plt.close(tmpfig)
    return 0

def clear_plot(event):
    global coords
    coords = []

    iplot_donor.clear()
    iplot_acc.clear()
    iplot_fret.clear()
    plot_efret.clear()
    
    draw_images()
    fig.canvas.draw_idle()

mosaic = [['image1','image2','image3'],
          ['plot1','plot1','plot1'],
          ['plot2','plot2','plot2'],
          ['plot3','plot3','plot3'],
          ['plot4','plot4','plot4']]
fig, ax = plt.subplot_mosaic(mosaic, gridspec_kw={'height_ratios': [4,1,1,1,1]})
fig.subplots_adjust(right=0.7, bottom=0.25)

donor_im = ax['image1']
acc_im = ax['image2']
fret_im = ax['image3']
iplot_donor = ax['plot1']
iplot_acc = ax['plot2']
iplot_fret = ax['plot3']
plot_efret = ax['plot4']

active_frame = 0
if len(sys.argv) == 2:
    fname = sys.argv[1]
    params = parameters.Parameters()
    params.name = fname[:-4]
    get_data(fname)
    draw_images()
else:
    fname = easygui.fileopenbox()
    params = parameters.Parameters()
    params.name = fname[:-4]
    get_data(fname)
    draw_images()

ck_ax = plt.axes([0.75, 0.7, 0.15, 0.075])
ck_ax.set_title("Chung-Kennedy?")
ck_radio = RadioButtons(ck_ax, ('Yes', 'No'), active=1)

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

slider_ax = fig.add_axes([0.2, 0.1, 0.5, 0.03])
frame_slider = Slider(
    ax=slider_ax,
    label='Frame',
    valmin=0,
    valmax=donor_data.num_frames,
    valinit=0,
)
frame_slider.on_changed(update)

frames_ax = plt.axes([0.75, 0.85, 0.15, 0.075])
frames_ax.set_title("Number of frames")
frames_box = TextBox(frames_ax, '', initial=donor_data.num_frames)
    
coords = []
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
fig.canvas.mpl_disconnect(cid)
