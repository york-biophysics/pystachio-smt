# =========================================================================
# TAB 2: DEDICATED MANUAL smFRET ANALYSIS TAB
# =========================================================================

import sys
import os
import glob
import csv
import subprocess
import traceback
import numpy as np
import tifffile as tf
import cv2
import matplotlib
matplotlib.use('QtAgg')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLineEdit,
                             QLabel, QRadioButton, QGroupBox, QFileDialog, 
                             QButtonGroup, QFormLayout, QMessageBox, QCheckBox,
                             QTabWidget, QGridLayout, QToolTip, QComboBox)
from PyQt6.QtCore import Qt

from pystackreg import StackReg

import images
import spots
import parameters
import postprocessing
import trajectories
import tracking
MODULES_LOADED = True


class HistogramWindow(QWidget):
    def __init__(self, parent_tab, e_vals, frame_idx, out_path):
        super().__init__()
        self.setWindowTitle(f"E_FRET Histogram - Frame {frame_idx}")
        self.resize(500, 400)
        
        layout = QVBoxLayout(self)
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        self.ax.hist(e_vals, bins=np.linspace(-0.2, 1.2, 29), color='green', alpha=0.7, edgecolor='black')
        self.ax.set_xlabel("FRET Efficiency (E)")
        self.ax.set_ylabel("Count")
        self.ax.set_title(f"E_FRET Distribution (Frame {frame_idx})")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.fig.tight_layout()
        
        try:
            self.fig.savefig(out_path, dpi=300)
            show_popup(self, "Histogram Saved", f"Successfully generated and saved histogram to:\n{out_path}")
        except Exception as e:
            show_popup(self, "Error Saving Histogram", f"Could not save histogram to disk: {e}", critical=True)

class StoichiometryWindow(QWidget):
    def __init__(self, parent_tab):
        super().__init__()
        self.parent_tab = parent_tab
        self.setWindowTitle("Select Stoichiometry (S vs E)")
        self.resize(700, 600)
        
        self.E_scatter = []
        self.S_scatter = []
        
        self.traj_E_avg = []
        self.traj_S_avg = []
        
        frames_to_avg = min(5, self.parent_tab.donor_data.num_frames) if self.parent_tab.donor_data else 0
        
        for i in range(len(self.parent_tab.trajs_donor)):
            d_trace = self.parent_tab.trajs_donor[i].intensity
            a_trace = self.parent_tab.trajs_acceptor[i].intensity
            f_trace = self.parent_tab.trajs_fret[i].intensity
            
            for f_idx in range(len(d_trace)):
                d, a, fr = d_trace[f_idx], a_trace[f_idx], f_trace[f_idx]
                e = fr / (d + fr) if (d + fr) > 0 else 0.0
                s = (d + fr) / (d + fr + a) if (d + fr + a) > 0 else 0.0
                self.E_scatter.append(e)
                self.S_scatter.append(s)
                
            d_avg = np.sum(d_trace[:frames_to_avg])
            a_avg = np.sum(a_trace[:frames_to_avg])
            fr_avg = np.sum(f_trace[:frames_to_avg])
            
            e_avg = fr_avg / (d_avg + fr_avg) if (d_avg + fr_avg) > 0 else 0.0
            s_avg = (d_avg + fr_avg) / (d_avg + fr_avg + a_avg) if (d_avg + fr_avg + a_avg) > 0 else 0.0
            
            self.traj_E_avg.append(e_avg)
            self.traj_S_avg.append(s_avg)
            
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        self.ax.scatter(self.E_scatter, self.S_scatter, c='blue', alpha=0.15, s=5, edgecolors='none', label='All Frames')
        self.ax.scatter(self.traj_E_avg, self.traj_S_avg, c='red', alpha=0.9, s=30, edgecolors='k', label=f'Trajectory Averages (First {min(5, self.parent_tab.donor_data.num_frames if self.parent_tab.donor_data else 0)} frames)')
        
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_xlabel("FRET Efficiency (E)")
        self.ax.set_ylabel("Stoichiometry (S)")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.legend(loc='upper right')
        
        self.rect = plt.Rectangle((0, 0), 1, 1, fill=False, color='red', linewidth=2)
        self.ax.add_patch(self.rect)
        
        controls = QGridLayout()
        
        self.e_min_sl = QSlider(Qt.Orientation.Horizontal); self.e_min_sl.setRange(0, 100); self.e_min_sl.setValue(0)
        self.e_max_sl = QSlider(Qt.Orientation.Horizontal); self.e_max_sl.setRange(0, 100); self.e_max_sl.setValue(100)
        self.s_min_sl = QSlider(Qt.Orientation.Horizontal); self.s_min_sl.setRange(0, 100); self.s_min_sl.setValue(0)
        self.s_max_sl = QSlider(Qt.Orientation.Horizontal); self.s_max_sl.setRange(0, 100); self.s_max_sl.setValue(100)
        
        self.e_min_sl.valueChanged.connect(self.update_rect)
        self.e_max_sl.valueChanged.connect(self.update_rect)
        self.s_min_sl.valueChanged.connect(self.update_rect)
        self.s_max_sl.valueChanged.connect(self.update_rect)
        
        controls.addWidget(QLabel("E min:"), 0, 0); controls.addWidget(self.e_min_sl, 0, 1)
        controls.addWidget(QLabel("E max:"), 1, 0); controls.addWidget(self.e_max_sl, 1, 1)
        controls.addWidget(QLabel("S min:"), 2, 0); controls.addWidget(self.s_min_sl, 2, 1)
        controls.addWidget(QLabel("S max:"), 3, 0); controls.addWidget(self.s_max_sl, 3, 1)
        
        layout.addLayout(controls)
        
        self.btn_confirm = QPushButton("Confirm selection")
        self.btn_confirm.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold;")
        self.btn_confirm.clicked.connect(self.confirm_selection)
        layout.addWidget(self.btn_confirm)
        
        self.update_rect()
        
    def update_rect(self):
        if self.e_min_sl.value() > self.e_max_sl.value(): self.e_min_sl.setValue(self.e_max_sl.value())
        if self.s_min_sl.value() > self.s_max_sl.value(): self.s_min_sl.setValue(self.s_max_sl.value())
        
        e_min, e_max = self.e_min_sl.value() / 100.0, self.e_max_sl.value() / 100.0
        s_min, s_max = self.s_min_sl.value() / 100.0, self.s_max_sl.value() / 100.0
        
        self.rect.set_xy((e_min, s_min))
        self.rect.set_width(e_max - e_min)
        self.rect.set_height(s_max - s_min)
        self.canvas.draw_idle()
        
    def confirm_selection(self):
        e_min, e_max = self.e_min_sl.value() / 100.0, self.e_max_sl.value() / 100.0
        s_min, s_max = self.s_min_sl.value() / 100.0, self.s_max_sl.value() / 100.0
        
        keep_indices = []
        for i, (e, s) in enumerate(zip(self.traj_E_avg, self.traj_S_avg)):
            if e_min <= e <= e_max and s_min <= s <= s_max:
                keep_indices.append(i)
                
        if not keep_indices:
            show_popup(self, "Selection Empty", "No spots fall within this selection. Try adjusting the sliders.", critical=True)
            return
            
        pt = self.parent_tab
        # Sync bounds locally so batch processor honors them later
        pt.e_min, pt.e_max = e_min, e_max
        pt.s_min, pt.s_max = s_min, s_max
        
        pt.coords = [pt.coords[i] for i in keep_indices]
        pt.trajs_donor = [pt.trajs_donor[i] for i in keep_indices]
        pt.trajs_acceptor = [pt.trajs_acceptor[i] for i in keep_indices]
        pt.trajs_fret = [pt.trajs_fret[i] for i in keep_indices]
        
        pt.refresh()
        self.close()

class SmFretTab(QWidget):
    def __init__(self, parent_suite=None):
        super().__init__()
        self.parent_suite = parent_suite
        # self.coords fundamentally stores ACCEPTOR/FRET channel spot coordinates
        self.coords = []
        self.donor_data = self.acceptor_data = self.fret_data = None
        self.trajs_donor, self.trajs_acceptor, self.trajs_fret = [], [], []
        self.params = parameters.Parameters()
        self.active_frame = 0
        self.registration_matrix = self.source_fname = None
        self.cb1 = self.cb2 = self.cb3 = None
        
        # Default Filter Boundaries
        self.e_min = 0.0
        self.e_max = 1.0
        self.s_min = 0.0
        self.s_max = 1.0

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        plot_layout = QVBoxLayout()
        self.fig = plt.figure(figsize=(14, 10))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        self.build_mosaic()
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame Index:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        slider_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0 / 0")
        slider_layout.addWidget(self.frame_label)
        plot_layout.addLayout(slider_layout)
        main_layout.addLayout(plot_layout, stretch=4)

        control_layout = QVBoxLayout()
        self.btn_pick_file = QPushButton("Load TIF Stack File")
        self.btn_pick_file.clicked.connect(self.pick_file)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_plot)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        
        self.btn_autodetect = QPushButton("Autodetect Spots")
        self.btn_autodetect.clicked.connect(self.autodetect_spots)
        
        self.btn_stoic = QPushButton("Select Stoichiometry")
        self.btn_stoic.clicked.connect(self.open_stoic_window)
        
        self.btn_batch_analyse = QPushButton("Analyse all")
        self.btn_batch_analyse.setStyleSheet("background-color: #842b5b; color: white; font-weight: bold;")
        self.btn_batch_analyse.clicked.connect(self.batch_analyse_all)

        self.chk_subdirs = QCheckBox("Include all subdirectories")
        self.chk_subdirs.setChecked(True)
        
        self.btn_hist = QPushButton("Show/Save E_FRET Histogram (Current Frame)")
        self.btn_hist.clicked.connect(self.show_histogram)
        
        control_layout.addWidget(self.btn_pick_file)
        control_layout.addWidget(self.btn_clear)
        control_layout.addWidget(self.btn_refresh)
        control_layout.addWidget(self.btn_autodetect)
        control_layout.addWidget(self.btn_stoic)
        control_layout.addWidget(self.btn_hist)
        control_layout.addWidget(self.btn_batch_analyse)
        control_layout.addWidget(self.chk_subdirs)

        temporal_group = QGroupBox("Temporal Deinterleaving (ALEX)")
        temporal_layout = QVBoxLayout()
        self.temporal_cluster = QButtonGroup(self)
        self.rb_temp_even = QRadioButton("Channel 1 / Even Frames First")
        self.rb_temp_odd = QRadioButton("Channel 2 / Odd Frames First")
        self.rb_temp_even.setChecked(True)
        self.temporal_cluster.addButton(self.rb_temp_even); self.temporal_cluster.addButton(self.rb_temp_odd)
        temporal_layout.addWidget(self.rb_temp_even); temporal_layout.addWidget(self.rb_temp_odd)
        temporal_group.setLayout(temporal_layout)
        control_layout.addWidget(temporal_group)

        spatial_group = QGroupBox("Spatial Orientation Map (smFRET)")
        spatial_layout = QVBoxLayout()
        self.spatial_cluster = QButtonGroup(self)
        self.rb_orient_norm = QRadioButton("Left: Donor | Right: FRET/Acceptor")
        self.rb_orient_inv = QRadioButton("Left: FRET/Acceptor | Right: Donor")
        self.rb_orient_norm.setChecked(True)
        self.spatial_cluster.addButton(self.rb_orient_norm); self.spatial_cluster.addButton(self.rb_orient_inv)
        spatial_layout.addWidget(self.rb_orient_norm); spatial_layout.addWidget(self.rb_orient_inv)
        spatial_group.setLayout(spatial_layout)
        control_layout.addWidget(spatial_group)
        
        self.rb_temp_even.toggled.connect(self.on_settings_changed)
        self.rb_orient_norm.toggled.connect(self.on_settings_changed)
        
        calib_group = QGroupBox("Calibration & Registration")
        calib_layout = QVBoxLayout()
        self.btn_registration = QPushButton("Calculate Registration Matrix")
        self.btn_registration.clicked.connect(self.parent_suite.run_registration if self.parent_suite else lambda: None)
        self.btn_clear_registration = QPushButton("Clear Registration Matrix")
        self.btn_clear_registration.clicked.connect(self.parent_suite.clear_registration if self.parent_suite else lambda: None)
        calib_layout.addWidget(self.btn_registration)
        calib_layout.addWidget(self.btn_clear_registration)
        calib_group.setLayout(calib_layout)
        control_layout.addWidget(calib_group)
        
        control_layout.addWidget(QLabel("Output name"))
        self.outname_box = QLineEdit("smFRET_output.dat")
        control_layout.addWidget(self.outname_box)
        self.btn_save = QPushButton("Export FRET Data")
        self.btn_save.clicked.connect(self.save_data)
        control_layout.addWidget(self.btn_save)
        
        ck_group = QGroupBox("Chung-Kennedy Filter Configuration")
        ck_layout = QVBoxLayout()
        self.ck_yes, self.ck_no = QRadioButton("Apply Filter View"), QRadioButton("Keep Raw Trajectory View")
        self.ck_yes.setChecked(True)
        ck_layout.addWidget(self.ck_yes); ck_layout.addWidget(self.ck_no)
        ck_group.setLayout(ck_layout)
        control_layout.addWidget(ck_group)
        self.ck_yes.toggled.connect(self.refresh)
        
        control_layout.addStretch()
        main_layout.addLayout(control_layout, stretch=1)

    def build_mosaic(self):
        self.fig.clear()
        mosaic = [['image1', 'image2', 'image3'],
                  ['plot1', 'plot1', 'plot1'],
                  ['plot2', 'plot2', 'plot2'],
                  ['plot3', 'plot3', 'plot3'],
                  ['plot4', 'plot4', 'plot4']]
        self.ax_dict = self.fig.subplot_mosaic(mosaic, gridspec_kw={'height_ratios': [5, 1, 1, 1, 1]})
        self.fig.subplots_adjust(left=0.06, right=0.92, top=0.96, bottom=0.06, hspace=0.35, wspace=0.15)
        self.donor_im, self.acc_im, self.fret_im = self.ax_dict['image1'], self.ax_dict['image2'], self.ax_dict['image3']
        self.iplot_donor, self.iplot_acceptor, self.iplot_fret = self.ax_dict['plot1'], self.ax_dict['plot2'], self.ax_dict['plot3']
        self.plot_efret = self.ax_dict['plot4']
        self.cb1 = self.cb2 = self.cb3 = None
        self.apply_plot_labels()
        self.canvas.draw_idle()

    def apply_plot_labels(self):
        self.iplot_donor.set_ylabel(r"Donor (x10$^3$)")
        self.iplot_acceptor.set_ylabel(r"Acceptor (x10$^3$)")
        self.iplot_fret.set_ylabel(r"FRET (x10$^3$)")
        self.plot_efret.set_ylabel("E_FRET")
        self.plot_efret.set_xlabel("Frame number")

    def on_settings_changed(self):
        if self.source_fname:
            self.get_data(self.source_fname)
            self.clear_plot(); self.draw_images()

    def pick_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open TIF Stack", filter="TIFF Files (*.tif *.tiff)")
        if not fname: return
        if self.parent_suite: self.parent_suite.reset_entire_suite()
        self.source_fname = fname
        self.params.name = fname[:-4]
        self.get_data(fname)
        self.update_ui_limits()
        self.draw_images()

    def _build_pystachio_image(self, numpy_array, fname):
        img = images.ImageData()
        img.read(fname, self.params)
        img.pixel_data = numpy_array
        img.num_frames = numpy_array.shape[0]
        img.frame_size = [numpy_array.shape[2], numpy_array.shape[1]]
        img.has_mask, img.mask_data = True, np.ones((numpy_array.shape[1], numpy_array.shape[2]))
        return img

    def get_data(self, fname):
        if not MODULES_LOADED: return
        try:
            raw_stack = tf.imread(fname)
            if raw_stack.ndim < 3: raw_stack = raw_stack[np.newaxis, :, :]
            mid_x = raw_stack.shape[2] // 2
            
            ch_left, ch_right = raw_stack[:, :, :mid_x], raw_stack[:, :, mid_x:]
            path_donor, path_acceptor_fret = (ch_left, ch_right) if self.rb_orient_norm.isChecked() else (ch_right, ch_left)
            
            if self.rb_temp_even.isChecked():
                donor, fret, acc = path_donor[0::2], path_acceptor_fret[0::2], path_acceptor_fret[1::2]
            else:
                donor, fret, acc = path_donor[1::2], path_acceptor_fret[1::2], path_acceptor_fret[0::2]
                
            min_len = min(len(donor), len(acc), len(fret))
            self.donor_data = self._build_pystachio_image(np.ascontiguousarray(donor[:min_len]), fname)
            self.acceptor_data = self._build_pystachio_image(np.ascontiguousarray(acc[:min_len]), fname)
            self.fret_data = self._build_pystachio_image(np.ascontiguousarray(fret[:min_len]), fname)
        except Exception as e:
            show_popup(self, "Dataset Import Failure", f"Failed to slice unified TIFF stack:\n{e}", critical=True)

    def update_ui_limits(self):
        if self.donor_data is not None:
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(self.donor_data.num_frames - 1)
            self.frame_slider.setValue(0); self.frame_slider.setEnabled(True)

    def on_frame_change(self, val):
        self.active_frame = val; self.draw_images()

    def draw_images(self):
        if self.donor_data is None: return
        self.donor_im.clear(); self.acc_im.clear(); self.fret_im.clear()
        f_idx = self.active_frame
        
        im1 = self.donor_im.imshow(self.donor_data[f_idx].as_image(), cmap='Greys_r')
        self.donor_im.set_xticks([]); self.donor_im.set_yticks([]); self.donor_im.set_title("Donor Channel Stack")
        im2 = self.acc_im.imshow(self.acceptor_data[f_idx].as_image(), cmap='Greys_r')
        self.acc_im.set_xticks([]); self.acc_im.set_yticks([]); self.acc_im.set_title("Acceptor Channel Stack")
        im3 = self.fret_im.imshow(self.fret_data[f_idx].as_image(), cmap='Greys_r')
        self.fret_im.set_xticks([]); self.fret_im.set_yticks([]); self.fret_im.set_title("FRET Channel Stack")
        
        self.cb1 = self.fig.colorbar(im1, ax=self.donor_im) if self.cb1 is None else self.cb1.update_normal(im1) or self.cb1
        self.cb2 = self.fig.colorbar(im2, ax=self.acc_im) if self.cb2 is None else self.cb2.update_normal(im2) or self.cb2
        self.cb3 = self.fig.colorbar(im3, ax=self.fret_im) if self.cb3 is None else self.cb3.update_normal(im3) or self.cb3
        
        self.frame_label.setText(f"Index: {f_idx} / {self.frame_slider.maximum()}")
        
        for coord in self.coords:
            ax_c, ay_c = float(coord[0]), float(coord[1])
            self.acc_im.scatter(ax_c, ay_c, marker='x', color='r')
            self.fret_im.scatter(ax_c, ay_c, marker='x', color='r')
            
            if self.registration_matrix is not None:
                try:
                    M_inv = cv2.invertAffineTransform(self.registration_matrix[:2, :])
                    M_inv_homo = np.vstack([M_inv, [0,0,1]])
                except:
                    M_inv_homo = np.eye(3)
                mapped = np.dot(M_inv_homo, np.array([ax_c, ay_c, 1.0]))[:2]
                dx, dy = float(mapped[0]), float(mapped[1])
            else:
                dx, dy = ax_c, ay_c
                
            self.donor_im.scatter(dx, dy, marker='x', color='r')
            
        self.canvas.draw_idle()

    def refresh(self):
        if self.donor_data is None: return
        self.draw_images()
        self.iplot_donor.clear(); self.iplot_acceptor.clear(); self.iplot_fret.clear(); self.plot_efret.clear()
        self.apply_plot_labels()
        num_frames = self.donor_data.num_frames
        
        for traj in range(len(self.trajs_donor)):
            tdonor, tacc, tfret = np.array(self.trajs_donor[traj].intensity), np.array(self.trajs_acceptor[traj].intensity), np.array(self.trajs_fret[traj].intensity)
            if self.ck_yes.isChecked():
                ck_donor = postprocessing.chung_kennedy_filter(tdonor, 5, 1)[0][:-1]
                ck_acc = postprocessing.chung_kennedy_filter(tacc, 5, 1)[0][:-1]
                ck_fret = postprocessing.chung_kennedy_filter(tfret, 5, 1)[0][:-1]
                self.iplot_donor.plot(ck_donor[:num_frames]/10**3); self.iplot_acceptor.plot(ck_acc[:num_frames]/10**3); self.iplot_fret.plot(ck_fret[:num_frames]/10**3)
                with np.errstate(divide='ignore', invalid='ignore'):
                    denom = ck_donor + ck_fret
                    self.plot_efret.plot(np.where(denom != 0, ck_fret / denom, 0)[:num_frames])
            else:
                self.iplot_donor.plot(tdonor[:num_frames]/10**3); self.iplot_acceptor.plot(tacc[:num_frames]/10**3); self.iplot_fret.plot(tfret[:num_frames]/10**3)
                with np.errstate(divide='ignore', invalid='ignore'):
                    denom_raw = tdonor + tfret
                    self.plot_efret.plot(np.where(denom_raw != 0, tfret / denom_raw, 0)[:num_frames])
                
        self.canvas.draw_idle()

    def clear_plot(self):
        self.coords, self.trajs_donor, self.trajs_acceptor, self.trajs_fret = [], [], [], []
        self.refresh()

    def _extract_trajectories(self, track_params):
        if not self.coords: return
        
        channel_spots_d, channel_spots_a, channel_spots_f = [], [], []
        donor_coords = []
        for c in self.coords:
            if self.registration_matrix is not None:
                try:
                    M_inv = cv2.invertAffineTransform(self.registration_matrix[:2, :])
                    M_inv_homo = np.vstack([M_inv, [0,0,1]])
                except:
                    M_inv_homo = np.eye(3)
                mapped = np.dot(M_inv_homo, np.array([c[0], c[1], 1.0]))[:2]
                mx, my = float(mapped[0]), float(mapped[1])
                if not np.isfinite(mx): mx = -10000.0
                if not np.isfinite(my): my = -10000.0
                mx = max(-10000.0, min(10000.0, mx))
                my = max(-10000.0, min(10000.0, my))
                donor_coords.append([mx, my])
            else:
                donor_coords.append([float(c[0]), float(c[1])])
                
        for f in range(self.donor_data.num_frames):
            sd = spots.Spots(frame=f)
            sd.set_positions(donor_coords); sd.num_spots = len(donor_coords)
            sd.get_spot_intensities(self.donor_data[f].as_image(), track_params)
            channel_spots_d.append(sd)
            
            sa = spots.Spots(frame=f)
            sa.set_positions(self.coords); sa.num_spots = len(self.coords)
            sa.get_spot_intensities(self.acceptor_data[f].as_image(), track_params)
            channel_spots_a.append(sa)
            
            sf = spots.Spots(frame=f)
            sf.set_positions(self.coords); sf.num_spots = len(self.coords)
            sf.get_spot_intensities(self.fret_data[f].as_image(), track_params)
            channel_spots_f.append(sf)
            
        self.trajs_donor = trajectories.build_trajectories(channel_spots_d, track_params)
        self.trajs_acceptor = trajectories.build_trajectories(channel_spots_a, track_params)
        self.trajs_fret = trajectories.build_trajectories(channel_spots_f, track_params)

    def onclick(self, event):
        if not self.donor_data or not MODULES_LOADED: return
        if event.inaxes not in [self.donor_im, self.acc_im, self.fret_im]: return
        mode = getattr(self.toolbar, 'mode', '')
        if hasattr(mode, 'value'): mode = mode.value
        if mode != "": return
            
        ix, iy = float(event.xdata), float(event.ydata)
        if event.inaxes == self.donor_im and self.registration_matrix is not None:
            mapped = np.dot(self.registration_matrix, np.array([ix, iy, 1.0]))[:2]
            ix, iy = float(mapped[0]), float(mapped[1])
            if not np.isfinite(ix): ix = -10000.0
            if not np.isfinite(iy): iy = -10000.0
            ix = max(-10000.0, min(10000.0, ix))
            iy = max(-10000.0, min(10000.0, iy))
        
        tmp_spots = spots.Spots(frame=0)
        tmp_spots.set_positions([[ix, iy]]); tmp_spots.num_spots = 1
        tmp_spots.refine_centres(self.acceptor_data[self.active_frame], self.params)
        if len(tmp_spots.positions) > 0:
            pos = tmp_spots.positions[0]
            self.coords.append([float(pos[0]), float(pos[1])])
        self._extract_trajectories(self.params)
        self.refresh()

    def autodetect_spots(self):
        if not self.donor_data or not MODULES_LOADED: return
        track_params = self.parent_suite.main_analysis_tab.params if self.parent_suite and hasattr(self.parent_suite, 'main_analysis_tab') else self.params
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.coords, self.trajs_donor, self.trajs_acceptor, self.trajs_fret = [], [], [], []
            spots_res = tracking.track_frame(self.acceptor_data[self.active_frame], self.active_frame, track_params)
            
            if not spots_res or len(spots_res.positions) == 0:
                QApplication.restoreOverrideCursor()
                show_popup(self, "Autodetect", "No spots found in the acceptor channel.")
                return

            for p in spots_res.positions:
                self.coords.append([float(p[0]), float(p[1])])

            self._extract_trajectories(track_params)
            self.refresh()
            QApplication.restoreOverrideCursor()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            show_popup(self, "Autodetect Failure", f"Error during spot autodetection:\n{traceback.format_exc()}", critical=True)

    def open_stoic_window(self):
        if not self.trajs_donor:
            show_popup(self, "Error", "No spots/trajectories available. Please manually click or autodetect spots first.", critical=True)
            return
        self.stoic_win = StoichiometryWindow(self)
        self.stoic_win.show()

    def show_histogram(self):
        if not self.trajs_donor:
            show_popup(self, "Error", "No spots/trajectories available to build histogram.", critical=True)
            return
            
        f_idx = self.active_frame
        e_vals = []
        for i in range(len(self.trajs_donor)):
            if f_idx < len(self.trajs_donor[i].intensity):
                d = self.trajs_donor[i].intensity[f_idx]
                f = self.trajs_fret[i].intensity[f_idx]
                e = f / (d + f) if (d + f) > 0 else 0.0
                e_vals.append(e)
                
        if not e_vals:
            show_popup(self, "Error", "No valid E_FRET values found in this frame.", critical=True)
            return
            
        rootname = self.params.name.rsplit('/', 1)[0] if self.params.name else os.path.dirname(self.source_fname)
        out_path = os.path.join(rootname, f"EFRET_hist_frame_{f_idx}.png")
        
        self.hist_win = HistogramWindow(self, e_vals, f_idx, out_path)
        self.hist_win.show()

    def save_data(self):
        if not self.params.name or not self.trajs_donor or not MODULES_LOADED: return
        out_file = self.params.name.rsplit('/', 1)[0] + '/' + self.outname_box.text()
        
        try:
            with open(out_file, "w") as f:
                f.write(f"{'Frame':<8} {'Traj_ID':<15} {'X_Coord':<10} {'Y_Coord':<10} {'Donor_Int':<15} {'Acceptor_Int':<15} {'FRET_Int':<15} {'E_FRET':<10}\n")
                for traj_id in range(len(self.trajs_donor)):
                    x, y = self.coords[traj_id]
                    for frame in range(self.donor_data.num_frames):
                        d_int, a_int, f_int = self.trajs_donor[traj_id].intensity[frame], self.trajs_acceptor[traj_id].intensity[frame], self.trajs_fret[traj_id].intensity[frame]
                        e_fret = f_int / (d_int + f_int) if (d_int + f_int) != 0 else 0.0
                        f.write(f"{frame:<8} {traj_id:<15} {x:<10.3f} {y:<10.3f} {d_int:<15.3f} {a_int:<15.3f} {f_int:<15.3f} {e_fret:<10.3f}\n")
            show_popup(self, "Export Success", f"Successfully exported analytical trajectory matrix to:\n{out_file}")
        except Exception as e:
            show_popup(self, "Save Failure", f"Failed to open/write analytical trajectory to disk space:\n{e}", critical=True)


    def batch_analyse_all(self):
        if not self.source_fname:
            show_popup(self, "Error", "Please load a file first to set the directory and deinterleaving configurations.", critical=True)
            return

        directory = os.path.dirname(self.source_fname)
        
        # --- NEW RECURSIVE SEARCH LOGIC ---
        if hasattr(self, 'chk_subdirs') and self.chk_subdirs.isChecked():
            tif_files = set(glob.glob(os.path.join(directory, "**", "*.tif"), recursive=True) + 
                            glob.glob(os.path.join(directory, "**", "*.tiff"), recursive=True))
        else:
            tif_files = set(glob.glob(os.path.join(directory, "*.tif")) + 
                            glob.glob(os.path.join(directory, "*.tiff")))
        # ----------------------------------
        
        skip_suffixes = ("_cropped.tif", "_cropped.tiff", "_left.tif", "_left.tiff", 
                         "_right.tif", "_right.tiff", "_donor.tif", "_donor.tiff", 
                         "_acceptor.tif", "_acceptor.tiff", "_fret.tif", "_fret.tiff")
        files_to_process = [f for f in tif_files if not f.endswith(skip_suffixes)]

        if not files_to_process:
            show_popup(self, "Info", "No valid un-processed .tif files found in this directory.")
            return

        track_params = self.parent_suite.main_analysis_tab.params if self.parent_suite and hasattr(self.parent_suite, 'main_analysis_tab') else self.params

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        orig_source_fname = self.source_fname
        orig_donor = self.donor_data
        orig_acceptor = self.acceptor_data
        orig_fret = self.fret_data
        orig_coords = self.coords
        orig_t_donor, orig_t_acc, orig_t_fret = self.trajs_donor, self.trajs_acceptor, self.trajs_fret

        out_raw_all = os.path.join(directory, "consolidated_smFRET_raw_all_frames.txt")
        out_filtered_all = os.path.join(directory, "consolidated_smFRET_filtered_all_frames.txt")
        out_filtered_first = os.path.join(directory, "consolidated_smFRET_filtered_first_frame.txt")
        out_summary_path = os.path.join(directory, "consolidated_smFRET_summary.txt")

        try:
            with open(out_raw_all, "w") as f_raw, \
                 open(out_filtered_all, "w") as f_filt_all, \
                 open(out_filtered_first, "w") as f_filt_first:
                 
                header = f"{'Image_Filename':<50} {'Traj_ID':>10} {'X_Coord':>10} {'Y_Coord':>10} {'Frame':>8} {'E_FRET':>10} {'S':>10} {'Donor_Int':>15} {'Acc_Int':>15} {'FRET_Int':>15}\n"
                f_raw.write(header)
                f_filt_all.write(header)
                f_filt_first.write(header)

                global_traj_id = 0

                for fname in files_to_process:
                    QApplication.processEvents()
                    try:
                        self.source_fname = fname
                        self.get_data(fname)
                    except Exception as e:
                        print(f"Skipping {fname} due to read error: {e}")
                        continue

                    if not self.acceptor_data: continue

                    all_coords = []
                    for f_idx in range(self.acceptor_data.num_frames):
                        QApplication.processEvents()
                        spots_res = tracking.track_frame(self.acceptor_data[f_idx], f_idx, track_params)
                        if spots_res and len(spots_res.positions) > 0:
                            for p in spots_res.positions:
                                pt = [float(p[0]), float(p[1])]
                                if not any(np.hypot(pt[0] - c[0], pt[1] - c[1]) < 2.0 for c in all_coords):
                                    all_coords.append(pt)

                    if not all_coords:
                        continue

                    self.coords = all_coords
                    self._extract_trajectories(track_params)

                    frames_to_avg = min(5, self.donor_data.num_frames)
                    
                    for i in range(len(self.trajs_donor)):
                        d_trace = self.trajs_donor[i].intensity
                        a_trace = self.trajs_acceptor[i].intensity
                        f_trace = self.trajs_fret[i].intensity
                        
                        d_avg = np.sum(d_trace[:frames_to_avg])
                        a_avg = np.sum(a_trace[:frames_to_avg])
                        fr_avg = np.sum(f_trace[:frames_to_avg])
                        
                        e_avg = fr_avg / (d_avg + fr_avg) if (d_avg + fr_avg) > 0 else 0.0
                        s_avg = (d_avg + fr_avg) / (d_avg + fr_avg + a_avg) if (d_avg + fr_avg + a_avg) > 0 else 0.0

                        passes_filter = (getattr(self, 'e_min', 0.0) <= e_avg <= getattr(self, 'e_max', 1.0)) and \
                                        (getattr(self, 's_min', 0.0) <= s_avg <= getattr(self, 's_max', 1.0))
                                        
                        x_coord, y_coord = self.coords[i]
                        base_fname = os.path.basename(fname)
                        
                        for f_idx in range(self.donor_data.num_frames):
                            d_val = d_trace[f_idx]
                            a_val = a_trace[f_idx]
                            fr_val = f_trace[f_idx]
                            
                            e_val = fr_val / (d_val + fr_val) if (d_val + fr_val) != 0 else 0.0
                            s_val = (d_val + fr_val) / (d_val + fr_val + a_val) if (d_val + fr_val + a_val) != 0 else 0.0

                            row_str = f"{base_fname:<50} {global_traj_id:>10} {x_coord:>10.3f} {y_coord:>10.3f} {f_idx:>8} {e_val:>10.3f} {s_val:>10.3f} {d_val:>15.3f} {a_val:>15.3f} {fr_val:>15.3f}\n"
                            
                            f_raw.write(row_str)
                            
                            if passes_filter:
                                f_filt_all.write(row_str)
                                if f_idx == 0:
                                    f_filt_first.write(row_str)
                                    
                        global_traj_id += 1

            with open(out_summary_path, "w") as f_sum:
                f_sum.write("smFRET Batch Analysis Summary\n")
                f_sum.write("=============================\n")
                f_sum.write(f"Processed {len(files_to_process)} files.\n\n")
                f_sum.write("Filters Applied:\n")
                f_sum.write(f"  E_FRET Range: {getattr(self, 'e_min', 0.0)} - {getattr(self, 'e_max', 1.0)}\n")
                f_sum.write(f"  Stoichiometry Range: {getattr(self, 's_min', 0.0)} - {getattr(self, 's_max', 1.0)}\n\n")
                f_sum.write("Tracking Parameters:\n")
                for k in dir(track_params):
                    if not k.startswith("__") and not callable(getattr(track_params, k)):
                        f_sum.write(f"  {k}: {getattr(track_params, k)}\n")
                f_sum.write("\nRegistration Matrix:\n")
                if self.registration_matrix is not None:
                    np.savetxt(f_sum, self.registration_matrix)
                else:
                    f_sum.write("  None loaded (Identity assumed).\n")

            QApplication.restoreOverrideCursor()
            show_popup(self, "Batch Analysis Completed", f"Successfully ran workflow and exported 3 formatted consolidated files (Raw, Filtered All, Filtered First) to:\n{directory}")

        except Exception as e:
            QApplication.restoreOverrideCursor()
            show_popup(self, "Pipeline Failure", f"Batch tracking runtime error:\n\n{traceback.format_exc()}", critical=True)

        finally:
            self.source_fname = orig_source_fname
            self.donor_data = orig_donor
            self.acceptor_data = orig_acceptor
            self.fret_data = orig_fret
            self.coords = orig_coords
            self.trajs_donor, self.trajs_acceptor, self.trajs_fret = orig_t_donor, orig_t_acc, orig_t_fret
            if self.source_fname:
                self.params.name = self.source_fname[:-4]
            self.refresh()

