# =========================================================================
# TAB 4: RESULTS VIEWER (POST-PROCESSING)
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


class ResultsViewerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        self.directory = ""
        self.data_store = {} 
        self.active_category = ""
        self.active_base_name = ""
        self.current_image_data = None
        self.plotted_tracks = [] 
        
        self.init_ui()

    def _create_scaled_slider_group(self, title, min_val, max_val, default_min, default_max, scale, is_int=False, single_slider=False):
        layout = QVBoxLayout()
        title_label = QLabel(f"<b>{title}</b>")
        layout.addWidget(title_label)
        
        row_min = QHBoxLayout()
        sl_min = QSlider(Qt.Orientation.Horizontal)
        sl_min.setRange(int(min_val * scale), int(max_val * scale))
        sl_min.setValue(int(default_min * scale))
        lbl_min = QLabel(f"{default_min}" if is_int else f"{default_min:.2f}")
        lbl_min.setFixedWidth(45)
        row_min.addWidget(QLabel("Min:" if not single_slider else "Value:"))
        row_min.addWidget(sl_min)
        row_min.addWidget(lbl_min)
        layout.addLayout(row_min)
        
        sl_max = None
        lbl_max = None
        if not single_slider:
            row_max = QHBoxLayout()
            sl_max = QSlider(Qt.Orientation.Horizontal)
            sl_max.setRange(int(min_val * scale), int(max_val * scale))
            sl_max.setValue(int(default_max * scale))
            lbl_max = QLabel(f"{default_max}" if is_int else f"{default_max:.2f}")
            lbl_max.setFixedWidth(45)
            row_max.addWidget(QLabel("Max:"))
            row_max.addWidget(sl_max)
            row_max.addWidget(lbl_max)
            layout.addLayout(row_max)
        
        def update_labels():
            if is_int:
                lbl_min.setText(f"{int(sl_min.value() / scale)}")
                if sl_max: lbl_max.setText(f"{int(sl_max.value() / scale)}")
            else:
                lbl_min.setText(f"{sl_min.value() / scale:.2f}")
                if sl_max: lbl_max.setText(f"{sl_max.value() / scale:.2f}")
                
        sl_min.valueChanged.connect(update_labels)
        if sl_max:
            sl_max.valueChanged.connect(update_labels)
        
        return layout, sl_min, sl_max, lbl_min, lbl_max

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        self.btn_dir = QPushButton("Select Results Directory")
        self.btn_dir.clicked.connect(self.load_directory)
        top_bar.addWidget(self.btn_dir)

        top_bar.addWidget(QLabel("Channel/Category:"))
        self.combo_category = QComboBox()
        top_bar.addWidget(self.combo_category)

        top_bar.addWidget(QLabel("Select File/Image (Single View):"))
        self.combo_image = QComboBox()
        top_bar.addWidget(self.combo_image)
        
        main_layout.addLayout(top_bar)

        middle_layout = QHBoxLayout()
        
        self.sub_tabs = QTabWidget()
        self.tab_single = QWidget()
        self.tab_cons = QWidget()
        self.sub_tabs.addTab(self.tab_single, "Single Image View")
        self.sub_tabs.addTab(self.tab_cons, "Consolidated View")
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.ax_img = self.axes[0, 0]
        self.ax_intensity = self.axes[0, 1]
        self.ax_stoich = self.axes[1, 0]
        self.ax_diff = self.axes[1, 1]
        self.fig.tight_layout()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self.tab_single)
        
        lay_single = QVBoxLayout(self.tab_single)
        lay_single.addWidget(self.toolbar)
        lay_single.addWidget(self.canvas)
        
        self.fig_cons, self.axes_cons = plt.subplots(1, 3, figsize=(12, 4))
        self.ax_cons_int = self.axes_cons[0]
        self.ax_cons_stoich = self.axes_cons[1]
        self.ax_cons_diff = self.axes_cons[2]
        self.fig_cons.tight_layout()
        self.canvas_cons = FigureCanvas(self.fig_cons)
        self.toolbar_cons = NavigationToolbar(self.canvas_cons, self.tab_cons)
        
        lay_cons = QVBoxLayout(self.tab_cons)
        lay_cons.addWidget(self.toolbar_cons)
        lay_cons.addWidget(self.canvas_cons)

        middle_layout.addWidget(self.sub_tabs, stretch=3)

        filter_box = QGroupBox("Filter Datasets")
        filter_layout = QVBoxLayout(filter_box)
        
        self.chk_req_stoich = QCheckBox("Only tracks with stoichiometry estimates")
        self.chk_req_diff = QCheckBox("Only tracks with diffusivity estimates")
        filter_layout.addWidget(self.chk_req_stoich)
        filter_layout.addWidget(self.chk_req_diff)
        
        snr_lay, self.sl_snr_min, _, _, _ = self._create_scaled_slider_group("SNR", 0, 1, 0, 1, 100, single_slider=True)
        filter_layout.addLayout(snr_lay)
        
        st_lay, self.sl_stoich_min, self.sl_stoich_max, _, _ = self._create_scaled_slider_group("Stoichiometry", 0, 100, 0, 100, 10)
        filter_layout.addLayout(st_lay)

        diff_lay, self.sl_diff_min, self.sl_diff_max, _, _ = self._create_scaled_slider_group("Diffusivity", -1, 10, -1, 10, 100)
        filter_layout.addLayout(diff_lay)
        
        sf_lay, self.sl_frame_min, self.sl_frame_max, self.lbl_f_min, self.lbl_f_max = self._create_scaled_slider_group("Trajectory Start Frame", 0, 10000, 0, 10000, 1, is_int=True)
        filter_layout.addLayout(sf_lay)
        
        filter_layout.addStretch()
        middle_layout.addWidget(filter_box, stretch=1)
        main_layout.addLayout(middle_layout)

        bottom_bar = QHBoxLayout()
        self.btn_export_current = QPushButton("Export Current Single Image Data & Plots")
        self.btn_export_current.clicked.connect(self.export_current_data)
        bottom_bar.addWidget(self.btn_export_current)

        self.btn_consolidate = QPushButton("Apply Filter & Export Consolidated Dataset")
        self.btn_consolidate.clicked.connect(self.consolidate_and_export)
        bottom_bar.addWidget(self.btn_consolidate)
        
        main_layout.addLayout(bottom_bar)

        self.combo_category.currentTextChanged.connect(self.change_category)
        self.combo_image.currentTextChanged.connect(self.change_image)
        self.sub_tabs.currentChanged.connect(self.update_plots)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        self.chk_req_stoich.stateChanged.connect(self.update_plots)
        self.chk_req_diff.stateChanged.connect(self.update_plots)
        self.sl_snr_min.valueChanged.connect(self.update_plots)
        self.sl_stoich_min.valueChanged.connect(self.update_plots)
        self.sl_stoich_max.valueChanged.connect(self.update_plots)
        self.sl_diff_min.valueChanged.connect(self.update_plots)
        self.sl_diff_max.valueChanged.connect(self.update_plots)
        self.sl_frame_min.valueChanged.connect(self.update_plots)
        self.sl_frame_max.valueChanged.connect(self.update_plots)

    def load_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Results Directory")
        if not dir_path:
            return
        self.directory = dir_path
        self.data_store.clear()
        
        traj_files = glob.glob(os.path.join(self.directory, "*_trajectories.tsv"))
        if not traj_files:
            show_popup(self, "No Files Found", "No tracking trajectories files (*_trajectories.tsv) found in this folder.", critical=True)
            return

        categories = ["donor", "acceptor", "fret", "left", "right"]
        
        for fpath in traj_files:
            fname = os.path.basename(fpath)
            base_name = fname.replace("_trajectories.tsv", "")
            
            matched_cat = "general"
            for cat in categories:
                if cat in fname.lower():
                    matched_cat = cat
                    break
            
            if matched_cat not in self.data_store:
                self.data_store[matched_cat] = {}
                
            self.data_store[matched_cat][base_name] = self.parse_file_package(fpath, base_name)
            
        self.combo_category.blockSignals(True)
        self.combo_category.clear()
        self.combo_category.addItems(list(self.data_store.keys()))
        self.combo_category.blockSignals(False)
        
        if self.combo_category.count() > 0:
            self.change_category(self.combo_category.currentText())

    def parse_file_package(self, traj_path, base_name):
        package = {"trajectories": [], "diff_coeff": {}, "intensity": [], "stoichiometry": {}}
        
        try:
            with open(traj_path, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    package["trajectories"].append({
                        "trajectory": int(row["trajectory"]),
                        "frame": int(row["frame"]),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "SNR": float(row["SNR"])
                    })
        except Exception as e:
            print(f"Error parsing trajectory file {traj_path}: {e}")

        diff_path = os.path.join(self.directory, f"{base_name}_diff_coeff_data.tsv")
        if os.path.exists(diff_path):
            try:
                with open(diff_path, 'r') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    for row in reader:
                        package["diff_coeff"][int(row["trajectory"])] = float(row["diffusion coefficient"])
            except Exception as e:
                print(f"Error parsing diffusion coefficients: {e}")

        int_path = os.path.join(self.directory, f"{base_name}_intensity_data.tsv")
        if os.path.exists(int_path):
            try:
                with open(int_path, 'r') as f:
                    package["intensity"] = [float(line.strip()) for line in f if line.strip()]
            except Exception as e:
                print(f"Error parsing intensity data: {e}")

        stoich_path = os.path.join(self.directory, f"{base_name}_stoichiometry_data.tsv")
        if os.path.exists(stoich_path):
            try:
                with open(stoich_path, 'r') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    for row in reader:
                        package["stoichiometry"][int(row["trajectory"])] = float(row["stoichiometry"])
            except Exception as e:
                print(f"Error parsing stoichiometry data: {e}")

        return package

    def change_category(self, cat):
        if not cat: return
        self.active_category = cat
        self.combo_image.blockSignals(True)
        self.combo_image.clear()
        self.combo_image.addItems(list(self.data_store[cat].keys()))
        self.combo_image.blockSignals(False)
        
        if self.combo_image.count() > 0:
            self.change_image(self.combo_image.currentText())
        else:
            self.update_plots()

    def change_image(self, base_name):
        if not base_name: return
        self.active_base_name = base_name
        
        package = self.data_store[self.active_category][self.active_base_name]
        if package["trajectories"]:
            max_f = max([p["frame"] for p in package["trajectories"]])
            self.sl_frame_min.blockSignals(True); self.sl_frame_max.blockSignals(True)
            self.sl_frame_min.setRange(0, max_f); self.sl_frame_max.setRange(0, max_f)
            self.sl_frame_min.setValue(0); self.sl_frame_max.setValue(max_f)
            self.lbl_f_min.setText("0"); self.lbl_f_max.setText(str(max_f))
            self.sl_frame_min.blockSignals(False); self.sl_frame_max.blockSignals(False)

        self.current_image_data = None
        for ext in [".tif", ".tiff", ".png"]:
            img_path = os.path.join(self.directory, f"{base_name}{ext}")
            if os.path.exists(img_path):
                try:
                    self.current_image_data = tf.imread(img_path)
                    if self.current_image_data.ndim >= 3:
                        self.current_image_data = self.current_image_data[0] 
                    break
                except Exception:
                    try:
                        self.current_image_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        break
                    except Exception: pass
        self.update_plots()

    def get_filter_limits(self):
        snr_min = self.sl_snr_min.value() / 100.0
        stoich_min, stoich_max = sorted([self.sl_stoich_min.value() / 10.0, self.sl_stoich_max.value() / 10.0])
        diff_min, diff_max = sorted([self.sl_diff_min.value() / 100.0, self.sl_diff_max.value() / 100.0])
        f_min, f_max = sorted([self.sl_frame_min.value(), self.sl_frame_max.value()])
        return snr_min, stoich_min, stoich_max, diff_min, diff_max, f_min, f_max

    def update_plots(self):
        if self.sub_tabs.currentIndex() == 0:
            self.update_single_view()
        else:
            self.update_consolidated_view()

    def update_single_view(self):
        if not self.active_category or not self.active_base_name: return
        
        package = self.data_store[self.active_category][self.active_base_name]
        snr_min, st_min, st_max, diff_min, diff_max, f_min, f_max = self.get_filter_limits()
        req_stoich = self.chk_req_stoich.isChecked()
        req_diff = self.chk_req_diff.isChecked()
        
        traj_groups = {}
        for pt in package["trajectories"]:
            t_id = pt["trajectory"]
            if t_id not in traj_groups: traj_groups[t_id] = []
            traj_groups[t_id].append(pt)
            
        for ax in self.axes.ravel(): ax.clear()
        self.plotted_tracks.clear()
        
        if self.current_image_data is not None:
            self.ax_img.imshow(self.current_image_data, cmap='gray', origin='lower')
        self.ax_img.set_title(f"Tracks Overlaid: {self.active_base_name}")

        valid_diffs, valid_stoichs = [], []
        cmap = plt.get_cmap("jet")
        
        for t_id, points in traj_groups.items():
            mean_snr = np.mean([p["SNR"] for p in points])
            start_frame = min([p["frame"] for p in points])
            
            diff = package["diff_coeff"].get(t_id, None)
            stoich = package["stoichiometry"].get(t_id, None)
            
            if req_diff and diff is None: continue
            if req_stoich and stoich is None: continue
            
            if mean_snr < snr_min: continue
            if not (f_min <= start_frame <= f_max): continue
            if diff is not None and not (diff_min <= diff <= diff_max): continue
            if stoich is not None and not (st_min <= stoich <= st_max): continue
            
            if diff is not None: valid_diffs.append(diff)
            if stoich is not None: valid_stoichs.append(stoich)
            
            points = sorted(points, key=lambda x: x["frame"])
            xs = [p["x"] for p in points]
            ys = [p["y"] for p in points]
            
            if diff is not None:
                norm_diff = (diff - diff_min) / max(1e-9, diff_max - diff_min)
                color = cmap(np.clip(norm_diff, 0, 1))
            else:
                color = (0.7, 0.7, 0.7, 0.8) 
            
            line, = self.ax_img.plot(xs, ys, color=color, alpha=0.8, linewidth=1.5)
            self.plotted_tracks.append({"line": line, "xs": xs, "ys": ys, "id": t_id, "diff": diff, "stoich": stoich})
            
        if package["intensity"]:
            self.ax_intensity.hist(package["intensity"], bins=30, color='green', alpha=0.7)
        self.ax_intensity.set_title("Overall Intensity Histogram")
        
        if valid_stoichs:
            self.ax_stoich.hist(valid_stoichs, bins=25, color='blue', alpha=0.7)
        self.ax_stoich.set_title("Filtered Stoichiometry")
        
        if valid_diffs:
            self.ax_diff.hist(valid_diffs, bins=25, color='red', alpha=0.7)
        self.ax_diff.set_title("Filtered Diffusivity")
        
        self.canvas.draw_idle()

    def update_consolidated_view(self):
        if not self.active_category: return
        
        snr_min, st_min, st_max, diff_min, diff_max, f_min, f_max = self.get_filter_limits()
        req_stoich = self.chk_req_stoich.isChecked()
        req_diff = self.chk_req_diff.isChecked()
        
        all_ints, all_diffs, all_stoichs = [], [], []
        
        for base_name, package in self.data_store[self.active_category].items():
            traj_groups = {}
            for pt in package["trajectories"]:
                t_id = pt["trajectory"]
                if t_id not in traj_groups: traj_groups[t_id] = []
                traj_groups[t_id].append(pt)
                
            for t_id, points in traj_groups.items():
                mean_snr = np.mean([p["SNR"] for p in points])
                start_frame = min([p["frame"] for p in points])
                diff = package["diff_coeff"].get(t_id, None)
                stoich = package["stoichiometry"].get(t_id, None)
                
                if req_diff and diff is None: continue
                if req_stoich and stoich is None: continue
                
                if mean_snr < snr_min: continue
                if not (f_min <= start_frame <= f_max): continue
                if diff is not None and not (diff_min <= diff <= diff_max): continue
                if stoich is not None and not (st_min <= stoich <= st_max): continue
                
                if diff is not None: all_diffs.append(diff)
                if stoich is not None: all_stoichs.append(stoich)
                
            if package["intensity"]:
                all_ints.extend(package["intensity"])
                
        self.ax_cons_int.clear()
        self.ax_cons_stoich.clear()
        self.ax_cons_diff.clear()
        
        if all_ints:
            self.ax_cons_int.hist(all_ints, bins=50, color='green', alpha=0.7)
        self.ax_cons_int.set_title(f"Consolidated Intensity ({len(all_ints)} pts)")
        
        if all_stoichs:
            self.ax_cons_stoich.hist(all_stoichs, bins=50, color='blue', alpha=0.7)
        self.ax_cons_stoich.set_title(f"Consolidated Stoichiometry ({len(all_stoichs)} tracks)")
        
        if all_diffs:
            self.ax_cons_diff.hist(all_diffs, bins=50, color='red', alpha=0.7)
        self.ax_cons_diff.set_title(f"Consolidated Diffusivity ({len(all_diffs)} tracks)")
        
        self.canvas_cons.draw_idle()

    def on_mouse_move(self, event):
        if self.sub_tabs.currentIndex() != 0 or event.inaxes != self.ax_img or not self.plotted_tracks:
            QToolTip.hideText()
            return
        
        mx, my = event.xdata, event.ydata
        closest_track = None
        min_dist = 5.0 
        
        for track in self.plotted_tracks:
            for tx, ty in zip(track["xs"], track["ys"]):
                dist = np.hypot(tx - mx, ty - my)
                if dist < min_dist:
                    min_dist = dist
                    closest_track = track
                    
        if closest_track:
            st_val = closest_track['stoich']
            df_val = closest_track['diff']
            st_str = f"{st_val:.2f}" if st_val is not None else "N/A"
            df_str = f"{df_val:.4f}" if df_val is not None else "N/A"
            msg = f"Track ID: {closest_track['id']}\nStoichiometry: {st_str}\nDiffusivity: {df_str}"
            QToolTip.showText(event.guiEvent.globalPosition().toPoint(), msg, self.canvas)
        else:
            QToolTip.hideText()

    def _write_filter_log(self, filepath):
        snr_min, st_min, st_max, diff_min, diff_max, f_min, f_max = self.get_filter_limits()
        with open(filepath, "w") as f:
            f.write(f"Start Frame range: {f_min} - {f_max}\n")
            f.write(f"Min SNR: {snr_min}\n")
            f.write(f"Stoichiometry range: {st_min} - {st_max}\n")
            f.write(f"Diffusivity range: {diff_min} - {diff_max}\n")
            f.write(f"Strict Stoich presence required: {self.chk_req_stoich.isChecked()}\n")
            f.write(f"Strict Diff presence required: {self.chk_req_diff.isChecked()}\n")

    def export_current_data(self):
        if not self.active_base_name: return
        package = self.data_store[self.active_category][self.active_base_name]
        out_root = os.path.join(self.directory, f"{self.active_base_name}_filtered")
        
        self._write_filter_log(f"{out_root}_filters_applied.txt")
        self.fig.savefig(f"{out_root}_plots.png", dpi=300)
        
        valid_tracks = {t["id"] for t in self.plotted_tracks}
        
        with open(f"{out_root}_trajectories.tsv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["trajectory", "frame", "x", "y", "SNR"], delimiter='\t', extrasaction='ignore')
            writer.writeheader()
            for pt in package["trajectories"]:
                if pt["trajectory"] in valid_tracks:
                    writer.writerow(pt)
                    
        with open(f"{out_root}_diff_coeff_data.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["trajectory", "diffusion coefficient"])
            for t_id in valid_tracks:
                if package["diff_coeff"].get(t_id) is not None:
                    writer.writerow([t_id, package["diff_coeff"][t_id]])
                
        with open(f"{out_root}_stoichiometry_data.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["trajectory", "stoichiometry"])
            for t_id in valid_tracks:
                if package["stoichiometry"].get(t_id) is not None:
                    writer.writerow([t_id, package["stoichiometry"][t_id]])
                
        show_popup(self, "Export Complete", f"Filtered files and visualization saved with suffix '_filtered' successfully.")

    def consolidate_and_export(self):
        if not self.active_category: return
        snr_min, st_min, st_max, diff_min, diff_max, f_min, f_max = self.get_filter_limits()
        req_stoich = self.chk_req_stoich.isChecked()
        req_diff = self.chk_req_diff.isChecked()
        
        c_traj, c_diff, c_stoich = [], [], []
        global_track_counter = 0
        
        self.sub_tabs.setCurrentIndex(1)
        self.update_consolidated_view()
        
        for base_name, package in self.data_store[self.active_category].items():
            traj_groups = {}
            for pt in package["trajectories"]:
                t_id = pt["trajectory"]
                if t_id not in traj_groups: traj_groups[t_id] = []
                traj_groups[t_id].append(pt)
            
            for t_id, points in traj_groups.items():
                mean_snr = np.mean([p["SNR"] for p in points])
                start_frame = min([p["frame"] for p in points])
                diff = package["diff_coeff"].get(t_id, None)
                stoich = package["stoichiometry"].get(t_id, None)
                
                if req_diff and diff is None: continue
                if req_stoich and stoich is None: continue
                if mean_snr < snr_min: continue
                if not (f_min <= start_frame <= f_max): continue
                if diff is not None and not (diff_min <= diff <= diff_max): continue
                if stoich is not None and not (st_min <= stoich <= st_max): continue
                    
                if diff is not None: c_diff.append([global_track_counter, diff])
                if stoich is not None: c_stoich.append([global_track_counter, stoich])
                
                for pt in points:
                    c_traj.append({
                        "trajectory": global_track_counter,
                        "frame": pt["frame"],
                        "x": pt["x"],
                        "y": pt["y"],
                        "SNR": pt["SNR"]
                    })
                global_track_counter += 1

        c_root = os.path.join(self.directory, f"consolidated_{self.active_category}")
        self.fig_cons.savefig(f"{c_root}_plots.png", dpi=300)
        
        with open(f"{c_root}_trajectories.tsv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["trajectory", "frame", "x", "y", "SNR"], delimiter='\t', extrasaction='ignore')
            writer.writeheader()
            writer.writerows(c_traj)
            
        with open(f"{c_root}_diff_coeff_data.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["trajectory", "diffusion coefficient"])
            writer.writerows(c_diff)
            
        with open(f"{c_root}_stoichiometry_data.tsv", "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["trajectory", "stoichiometry"])
            writer.writerows(c_stoich)
            
        self._write_filter_log(f"{c_root}_filters_applied.txt")
        show_popup(self, "Consolidation Success", f"Successfully generated compiled/consolidated datasets for the '{self.active_category}' channel.")

