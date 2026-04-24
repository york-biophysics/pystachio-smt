# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from keras.models import load_model
import tensorflow as tf
import torch
import glob
import shutil
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
from scipy import optimize, ndimage
import scipy.io as sio
from skimage import io, measure
from skimage import transform as ski_transform
from skimage.util import img_as_float64, img_as_uint
from PIL import Image
from pystackreg import StackReg
from roifile import ImagejRoi
import pandas as pd
import json
import tifffile
import traceback
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import plotly.express as px

class FileHandler:
    """Utility class for managing files and directory structures."""
    @staticmethod
    def delete_folders(pattern: str):
        matches = glob.glob(pattern)
        for folder in [d for d in matches if os.path.isdir(d)]:
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(f"Error deleting {folder}: {e}", flush=True)

    @staticmethod
    def save_parameters(args_dict: dict, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/parameters.txt', 'w') as f:
            for k, v in args_dict.items(): 
                f.write(f"{k}: {v}\n")

class MathUtils:
    @staticmethod
    def exp_1(x, A, k, c): return A * np.exp(-k * x) + c
    @staticmethod
    def exp_2(x, A, k, A1, k1, c): return A * np.exp(-k * x) + A1 * np.exp(-1 * (k1) * x) + c
    @staticmethod
    def exp_3(x, A, k, A1, k1, A2, k2, c): return A * np.exp(-k * x) + A1 * np.exp(-1 * (k1) * x) + A2 * np.exp(-1 * (k2) * x) + c
    @staticmethod
    def chisq(ydata, ymodel, err): return np.sum((np.asarray(ydata) - np.asarray(ymodel))**2 / err)

class IntensityAnalyzer:
    """Handles plotting of intensity traces and fitting to exponential decay models."""
    
    @staticmethod
    def plot_intensity(intensity_data, channel, index, frames, save_dir):
        intensity_data_x = np.asarray(intensity_data[0])
        intensity_data_y = np.asarray(intensity_data[1])
        
        # 1. Plot the full raw intensity trace
        fig = plt.figure(figsize=(7, 5))
        ax = fig.subplots()
        ax.plot(intensity_data_x[:frames], intensity_data_y[:frames], label=f'{channel}')
        ax.legend(fontsize=12)
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Intensity (Counts)")
        plt.savefig(f'{save_dir}/Int_intensity_{channel}_obj_{index}.png')
        plt.close()

        # 2. Plot the first 50 frames and fit exponential decay models
        fig = plt.figure(figsize=(7, 5))
        ax = fig.subplots()
        
        # Safely determine frame ranges
        fit_frames = min(50, len(intensity_data_x))
        x_fit_range = intensity_data_x[:fit_frames]
        y_fit_range = intensity_data_y[:fit_frames]
        
        ax.plot(x_fit_range, y_fit_range, label=f'{channel}')
        
        fit_quality = []
        baseline_index = min(50, len(intensity_data_y) - 1)
        baseline = intensity_data_y[baseline_index]
        
        guess_frames = min(10, len(intensity_data_x))
        x_, y_ = intensity_data_x[:guess_frames], intensity_data_y[:guess_frames]

        try:
            fit_exp_1 = lambda x, A, k: MathUtils.exp_1(x, A, k, baseline)
            params, cov = optimize.curve_fit(fit_exp_1, x_, y_, p0=(max(y_), 1))
            model_1 = MathUtils.exp_1(x_fit_range, params[0], params[1], baseline)
            rchisq_1 = MathUtils.chisq(y_fit_range, model_1, model_1) / max(1, (len(model_1) - 1 - 2))
            ax.plot(x_fit_range, model_1, linestyle="--", label="Single exp")
            fit_quality.append([params, rchisq_1])
        except Exception:
            print(f"Cell {index} ({channel}): Cannot fit to 1 exp", flush=True)
            fit_quality.append(0)
            
        try:
            fit_exp_2 = lambda x, A, k, A1, k1: MathUtils.exp_2(x, A, k, A1, k1, baseline)
            params, cov = optimize.curve_fit(fit_exp_2, x_, y_, p0=(1e5, 1, 1e5, 1))
            model_2 = MathUtils.exp_2(x_fit_range, params[0], params[1], params[2], params[3], baseline)
            rchisq_2 = MathUtils.chisq(y_fit_range, model_2, model_2) / max(1, (len(model_2) - 1 - 4))
            ax.plot(x_fit_range, model_2, linestyle="--", label="Double exp")
            fit_quality.append([params, rchisq_2])
        except Exception:
            print(f"Cell {index} ({channel}): Cannot fit to 2 exp", flush=True)
            fit_quality.append(0)

        try:    
            fit_exp_3 = lambda x, A, k, A1, k1, A2, k2: MathUtils.exp_3(x, A, k, A1, k1, A2, k2, baseline)
            params, cov = optimize.curve_fit(fit_exp_3, x_, y_, p0=(0.75e5, 1, 0.75e5, 1, 0.75e5, 1))
            model_3 = MathUtils.exp_3(x_fit_range, params[0], params[1], params[2], params[3], params[4], params[5], baseline)
            rchisq_3 = MathUtils.chisq(y_fit_range, model_3, model_3) / max(1, (len(model_3) - 1 - 6))
            ax.plot(x_fit_range, model_3, linestyle="--", label="Triple exp")
            fit_quality.append([params, rchisq_3])    
        except Exception:
            print(f"Cell {index} ({channel}): Cannot fit to 3 exp", flush=True)
            fit_quality.append(0)

        ax.legend(fontsize=8)
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Intensity (Counts)")
        plt.savefig(f'{save_dir}/Int_intensity_fit_{channel}_obj_{index}.png')
        plt.close()
        
        return fit_quality

class ImageProcessor:
    @staticmethod
    def read_roi(roi_path: str, roi_channel: str, current_channel: str, img_height: int, img_width: int, is_split_view: bool = True):
        # Calculate single channel width based on physical split, not user parameters
        half_width = img_width // 2 if is_split_view else img_width
        
        if not os.path.exists(roi_path):
            return [0, img_height, 0, half_width]
            
        try:
            roi = ImagejRoi.fromfile(roi_path)
            y1, x1, y2, x2 = int(roi.top), int(roi.left), int(roi.bottom), int(roi.right)
            
            # Convert any global coordinates to local single-channel coordinates (0 to 1024)
            if is_split_view:
                if x1 >= half_width: x1 -= half_width
                if x2 > half_width:  x2 -= half_width
                
            # Clamp the ROI so it never exceeds the boundaries of a single channel
            y1 = max(0, min(y1, img_height))
            y2 = max(0, min(y2, img_height))
            x1 = max(0, min(x1, half_width))
            x2 = max(0, min(x2, half_width))
            
            return [y1, y2, x1, x2]
            
        except Exception as e:
            print(f"Error reading ROI file: {e}", flush=True)
            return [0, img_height, 0, half_width]

    @staticmethod
    def make_patches(input_image, inv_img):
        # 1. Pad the image with the mean background (like the old script), NOT zeros
        img = input_image.copy().astype(np.float64)
        bkg_1 = np.mean(img)
        
        # Calculate padding needed to reach multiples of 256
        h, w = img.shape
        pad_h = (256 - (h % 256)) % 256
        pad_w = (256 - (w % 256)) % 256
        
        # Apply the background padding
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=bkg_1)

        patches = []
        num_h_patches = img.shape[0] // 256
        num_w_patches = img.shape[1] // 256
        
        for i in range(num_h_patches):
            for j in range(num_w_patches):
                y, x = i * 256, j * 256
                patch = img[y:y+256, x:x+256]
                
                # Apply inversion if requested
                if str(inv_img).lower() == "true":
                    patch = np.amax(patch) - patch
                    bkg_2 = np.mean(patch)
                    patch = patch - bkg_2 + bkg_1
                
                # 2. CRITICAL FIX: Normalize the individual patch by its max value
                if patch.max() > 0:
                    patch = patch / patch.max()
                    
                patches.append(patch)
                
        return patches, num_h_patches, num_w_patches

    @staticmethod
    def stitch_patches(predictions, original_shape, num_h_patches, num_w_patches):
        stitched_image = np.zeros((num_h_patches * 256, num_w_patches * 256))
        index = 0
        for i in range(num_h_patches):
            for j in range(num_w_patches):
                y1, x1 = i * 256, j * 256
                stitched_image[y1:y1+256, x1:x1+256] = predictions[index]
                index += 1
        return stitched_image[:original_shape[0], :original_shape[1]].astype(np.uint8)

    @staticmethod
    def apply_watershed_threshold(img_for_masking, dist_multiplier=0.3):
        img_float = img_for_masking.astype(np.float32)
        p_low, p_high = np.percentile(img_float, (1.0, 99.5))
        img_clipped = np.clip(img_float, p_low, p_high)
        
        img_uint8 = cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_uint8)
        
        blurred = cv2.medianBlur(img_clahe, 5)
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            45, 
            -2  
        )
        
        #_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        sure_bg = cv2.dilate(opening, kernel, iterations=2)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, dist_multiplier * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        final_mask = np.zeros_like(img_uint8)
        final_mask[markers > 1] = 255
        
        return ndimage.binary_fill_holes(final_mask).astype(np.uint8) * 255

    @staticmethod
    def generate_dual_masks(mask, target_channel):
        """
        Generates an inverted background mask for the non-target channel 
        when processing dual-channel videos.
        """
        # 1. Generate the inverted mask with blur and morphology
        blur = cv2.GaussianBlur(mask, (25, 25), 0)
        ret, inv_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        morph_kernel = np.ones((5, 5), np.uint8)
        inv_mask = cv2.dilate(inv_mask, morph_kernel, iterations=2)
        inv_mask = cv2.erode(inv_mask, morph_kernel, iterations=1)

        # 2. Assign the standard mask and inv_mask to the correct channels
        if target_channel == 'L':
            return mask, inv_mask
        elif target_channel == 'R':
            return inv_mask, mask
        else:
            return mask, mask

class VideoProcessor:
    @staticmethod
    def process_video(video_path, start_frame, end_frame, roi_channel, roi_file, save_dir, num_channels, channel, ALEX):
        frames = io.imread(video_path)
        
        # Handle cases where the TIF might load as 2D (single frame)
        if len(np.shape(frames)) == 2:
            frames = np.expand_dims(frames, axis=0)
            
        frames = frames[start_frame:end_frame]
            
        raw_shape = np.shape(frames)
        height, width = raw_shape[1], raw_shape[2]
        
        # AUTO-DETECT SPLIT CAMERA: Even if the user passes num_channels=1, if it's 2048 wide, WE SPLIT IT.
        is_split_view = width > 1500
        half_width = width // 2 if is_split_view else width
        
        y1, y2, x1, x2 = ImageProcessor.read_roi(
            roi_file, roi_channel, channel, height, width, is_split_view
        )
        
        if is_split_view:
            if str(ALEX).lower() == "false":
                L_full = frames[:, :, :half_width]
                R_full = frames[:, :, half_width:]
            else:
                L_full = frames[::2, :, :half_width]
                R_full = frames[1::2, :, half_width:]
                
            L_channel = L_full[:, y1:y2, x1:x2]
            R_channel = R_full[:, y1:y2, x1:x2]
            
            # If they specifically requested 1 channel but it's a split image, just return the requested one
            if int(num_channels) == 1:
                if channel == "L": return (L_channel, None, end_frame - start_frame, raw_shape)
                else: return (None, R_channel, end_frame - start_frame, raw_shape)
            else:
                return L_channel, R_channel, end_frame - start_frame, raw_shape
        else:
            # Single camera physical view
            channel_data = frames[:, y1:y2, x1:x2]
            if channel == "L": return (channel_data, None, end_frame - start_frame, raw_shape)
            else: return (None, channel_data, end_frame - start_frame, raw_shape)

class BeadRegistrar:
    """Handles bead-based spatial registration using the best NCC score."""
    
    @staticmethod
    def calculate_ncc(image_a, image_b):
        image_a = image_a.astype(np.float64)
        image_b = image_b.astype(np.float64)
        mean_a = np.mean(image_a)
        mean_b = np.mean(image_b)
        centered_a = image_a - mean_a
        centered_b = image_b - mean_b
        numerator = np.sum(centered_a * centered_b)
        denominator = np.sqrt(np.sum(centered_a ** 2) * np.sum(centered_b ** 2))
        return 0 if denominator == 0 else numerator / denominator

    @staticmethod
    def normalize_channels(i1, i2):
        i1 = i1.astype(np.float64)
        i2 = i2.astype(np.float64)
        m1, m2 = np.amax(i1), np.amin(i1)
        if m1 != m2: i1 = 1024 * (i1 - m2) / (m1 - m2)
        m1, m2 = np.amax(i2), np.amin(i2)
        if m1 != m2: i2 = 1024 * (i2 - m2) / (m1 - m2)

        m, n = np.amax(i2), np.amax(i1)
        if m != 0: i2 = n * i2 / m
        if np.amax(i1) != 0: i1 = 255 * i1 / np.amax(i1)
        if np.amax(i2) != 0: i2 = 255 * i2 / np.amax(i2)
            
        return i1, i2

    def save_registration_plot(self, static_img, moving_img, registered_img, ncc, name, save_dir):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].imshow((static_img + moving_img) / 2, cmap='gray')
        axs[1].imshow(registered_img, cmap='gray')
        axs[2].imshow((static_img + registered_img) / 2, cmap='gray')
        safe_name = os.path.basename(os.path.normpath(name))
        plt.tight_layout()
        plt.savefig(f"{save_dir}/Registered_overlay_{safe_name}.png")
        plt.close()

    def calculate_transformation(self, bead_prefix: str, save_dir: str, channel: str):
        bead_folders = glob.glob(f"{bead_prefix}_*")
        if not bead_folders: return None

        ncc_scores = []
        sr = StackReg(StackReg.RIGID_BODY)
        summed_img = None
        
        # Track NCCs for the line plot
        ncc_before_list = []
        ncc_after_list = []
        
        import pandas as pd

        for idx, folder in enumerate(bead_folders):
            files = glob.glob(f"{folder}/*.ome.tif") + glob.glob(f"{folder}/*.tif")
            if not files: continue
            beads = io.imread(files[0])
            if beads.ndim == 3: beads = np.mean(beads, axis=0)
                
            beads_float = img_as_float64(beads)
            if summed_img is not None: summed_img += beads_float
            else: summed_img = beads_float.copy()
                
            width = beads.shape[1]
            L, R = beads[:, :width//2], beads[:, width//2:]
            
            # Store exact copies for individual plotting
            if channel.upper() == "L":
                i1, i2 = L.copy(), R.copy()
            else:
                i1, i2 = R.copy(), L.copy()

            L_norm, R_norm = self.normalize_channels(L, R)

            if channel.upper() == "L":
                tmat = sr.register(L_norm, R_norm)
            else:
                tmat = sr.register(R_norm, L_norm)
                
            # --- INDIVIDUAL BEAD OVERLAY LOGIC ---
            m1, m2 = np.amax(i1), np.amin(i1)
            if m1 != m2: i1 = 1024 * (i1 - m2) / (m1 - m2)
            m1, m2 = np.amax(i2), np.amin(i2)
            if m1 != m2: i2 = 1024 * (i2 - m2) / (m1 - m2)

            m, n = np.amax(i2), np.amax(i1)
            if m != 0: i2 = n * i2 / m
            if np.amax(i1) != 0: i1 = 255 * i1 / np.amax(i1)
            if np.amax(i2) != 0: i2 = 255 * i2 / np.amax(i2)
            
            out = sr.transform(i2, tmat)
            
            # Calculate NCCs
            ncc_before = self.calculate_ncc(i1, i2)
            ncc_after = self.calculate_ncc(i1, out)
            ncc_before_list.append(ncc_before)
            ncc_after_list.append(ncc_after)
            
            # Fix background padding for plot
            bkg_1 = np.mean(i2)
            out[out == 0] = bkg_1
            
            # 1. Save Individual Overlay (Registered_overlay_beads_1.png, etc)
            fig, axs = plt.subplots(3, 1, figsize=(10, 15))
            axs[0].imshow((i1 + i2) / 2, 'gray')
            axs[0].set_title(f'Original Images - beads_{idx+1}')
            axs[0].set_xticks([]); axs[0].set_yticks([])

            axs[1].imshow(out, 'gray')
            axs[1].set_title('Registered Image')
            axs[1].set_xticks([]); axs[1].set_yticks([])

            axs[2].imshow(((i1 + out) / 2), cmap='gray')
            axs[2].set_title(f'Overlay - NCC = {ncc_after}')
            axs[2].set_xticks([]); axs[2].set_yticks([])
            plt.tight_layout()
            plt.savefig(f"{save_dir}/Registered_overlay_beads_{idx+1}.png")
            plt.close()
            # -------------------------------------
            
            ncc_scores.append({'folder': folder, 'ncc': ncc_after, 'matrix': tmat})

        best_entry = max(ncc_scores, key=lambda x: x['ncc'])
        tmats = best_entry['matrix']
        np.save(f"{save_dir}/transformation_matrix.npy", tmats)

        # 2. Save Summed Beads TIF 
        tifffile.imwrite(f"{save_dir}/summed_beads.tif", summed_img.astype(np.float32), imagej=True)
        
        # Split summed image
        split_width = summed_img.shape[1]
        L_sum = summed_img[:, :split_width//2]
        R_sum = summed_img[:, split_width//2:]
        
        if channel.upper() == "L":
            i1, i2 = L_sum, R_sum
        else:
            i1, i2 = R_sum, L_sum
            
        m1, m2 = np.amax(i1), np.amin(i1)
        if m1 != m2: i1 = 1024 * (i1 - m2) / (m1 - m2)
        m1, m2 = np.amax(i2), np.amin(i2)
        if m1 != m2: i2 = 1024 * (i2 - m2) / (m1 - m2)

        m, n = np.amax(i2), np.amax(i1)
        if m != 0: i2 = n * i2 / m
        if np.amax(i1) != 0: i1 = 255 * i1 / np.amax(i1)
        if np.amax(i2) != 0: i2 = 255 * i2 / np.amax(i2)
        
        out = sr.transform(i2, tmats)
        ncc_score = best_entry['ncc']
        
        bkg_1 = np.mean(i2)
        out[out == 0] = bkg_1
        
        # 3. Save Summed Overlay (Registered_overlay_summed_beads.png)
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].imshow((i1 + i2) / 2, 'gray')
        axs[0].set_title('Original Images - summed_beads')
        axs[0].set_xticks([]); axs[0].set_yticks([])

        axs[1].imshow(out, 'gray')
        axs[1].set_title('Registered Image')
        axs[1].set_xticks([]); axs[1].set_yticks([])

        axs[2].imshow(((i1 + out) / 2), cmap='gray')
        axs[2].set_title(f'Overlay - NCC = {ncc_score}')
        axs[2].set_xticks([]); axs[2].set_yticks([])
        plt.tight_layout()
        plt.savefig(f"{save_dir}/Registered_overlay_summed_beads.png")
        plt.close()
        
        # 4. Save Registration CSV
        df_reg = pd.DataFrame.from_dict({'sum': {'ncc_score': ncc_score, 'transformation_matrix': tmats.tolist()}}, orient='index')
        df_reg.to_csv(f"{save_dir}/registration.csv")

        # 5. Save ncc_scores.csv and ncc_scores.png
        df_ncc = pd.DataFrame({
            'Bead_Folder': range(1, len(ncc_before_list) + 1),
            'NCC_Before': ncc_before_list,
            'NCC_After': ncc_after_list
        })
        df_ncc.to_csv(f"{save_dir}/ncc_scores.csv", index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(df_ncc['Bead_Folder'], df_ncc['NCC_Before'], label='Before Registration', color='red', marker='o')
        plt.plot(df_ncc['Bead_Folder'], df_ncc['NCC_After'], label='After Registration', color='blue', marker='o')
        plt.title('Normalized Cross-Correlation (NCC) Scores')
        plt.xlabel('Bead Folder / Frame')
        plt.ylabel('NCC Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/ncc_scores.png", dpi=300, bbox_inches='tight')
        plt.close()

        return tmats
    
    
    def interactive_manual_registration(self, image_path: str, save_dir: str, num_pairs: int = 10):
        """
        Interactive point-picking for manual affine registration.
        """
        print(f"Loading image for manual registration: {image_path}", flush=True)
        # Load and handle potential 3D stacks
        img = io.imread(image_path)
        if img.ndim == 3: 
            img = np.mean(img, axis=0) 
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Normalize for display so 16-bit images don't appear pure black
        img_display = img.astype(float)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        ax.imshow(img_display, cmap='gray')
        
        ax.set_title(f"Click {num_pairs} pairs of beads.\n"
                     f"Pattern: [Ch1 Bead, Ch2 Bead], [Ch1 Bead, Ch2 Bead]...\n"
                     f"Middle-click to undo, Right-click to stop early.")
        plt.axis('image')
        
        print(f"Waiting for user to select {num_pairs * 2} points on the image...", flush=True)
        pts = plt.ginput(num_pairs * 2, timeout=-1) 
        plt.close(fig)
        
        if len(pts) % 2 != 0:
            print("Warning: Odd number of points selected. Discarding the last point.", flush=True)
            pts = pts[:-1]
            
        if len(pts) < 6:
            raise ValueError("Error: An affine transform requires at least 3 pairs (6 points) to compute.")
            
        pts = np.array(pts)
        pts_ch1 = pts[0::2] # Reference
        pts_ch2 = pts[1::2] # Moving
        
        # Estimate Affine Transform
        tform = ski_transform.estimate_transform('affine', src=pts_ch2, dst=pts_ch1)
        
        print("\nEstimated Affine Transformation Matrix:", flush=True)
        print(tform.params, flush=True)
        
        # 1. Save as .npy to match the existing Python pipeline's format
        out_npy = os.path.join(save_dir, "transformation_matrix.npy")
        np.save(out_npy, tform.params)
        
        # 2. Save as .mat for MATLAB ADEMScode compatibility
        out_mat = os.path.join(save_dir, "channel_transform.mat")
        mat_dict = {
            'T_matrix': tform.params.T, # Transpose for MATLAB
            'tform_type': 'affine',
            'generated_by': 'Python scikit-image manual registration'
        }
        sio.savemat(out_mat, mat_dict)
        
        print(f"\nSaved transformation matrix to {out_npy} and {out_mat}", flush=True)
        return tform.params

class AnalysisPipeline:
    def __init__(self, params):
        """
        Integrated PySTACHIO Initializer.
        This replaces the old argparse-based __init__.
        """
        self.params = params
        
        # --- 1. Map Universal PySTACHIO Params to Internal Names ---
        self.pxsize = params.pixel_size        # mapped from pixel_size
        self.ALEX = params.ALEX                # mapped from ALEX
        self.save_dir = params.name            # Use the seed 'name' as the directory
        self.channel = params.use_channel      # mapped from use_channel
        self.num_frames = None if params.num_frames == 0 else params.num_frames

        # --- 2. Map Preprocessing-Specific Params ---
        self.video_path = params.video_path
        self.bead_path = params.bead_path
        self.bf_path = params.bf_path
        self.tmats_path = params.tmats_path
        self.manual_registration = params.manual_registration
        self.manual_pairs = params.manual_pairs
        self.roi_channel = params.roi_channel
        self.roi_file = params.roi_file
        self.mask_prefix = params.mask_prefix
        self.overwrite = params.overwrite
        self.use_otsu = params.use_otsu
        self.cell_fitting = params.cell_fitting
        self.area_filter = int(params.area_filter)
        self.inv_bf = params.inv_bf

        # --- 3. Model Loading Logic (Your original logic, upgraded) ---
        self.model = None
        
        # Note: We use params.mask_type and params.model here
        if params.mask_type in ["AI", "BF", "FL_AI"]:
            print(f"Loading Model: {params.model}", flush=True)
            
            if params.model_type == "omnitorch":
                print("Omnipose execution deferred to external RunOmnipose script.", flush=True)
                
            elif params.model_type in ["unet", "keras"]:
                print("Loading Keras/U-Net model...", flush=True)
                # Ensure you still have the load_model import at the top of the file
                self.model = load_model(params.model, compile=False, safe_mode=False)
                
            elif params.model_type == "pytorch":
                print("Loading standard PyTorch model...", flush=True)
                self.model = torch.load(params.model, map_location=torch.device('cpu'))
                self.model.eval() 

        # --- 4. Setup Directory ---
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            
    def process_brightfield(self, raw_shape, sr, tmats):
        bf_sum = None
        bf_cropped = None
        width = raw_shape[2]
        
        # ALWAYS process BF if a path is given, regardless of mask_type
        if self.args.bf_path and os.path.exists(self.args.bf_path):
            bf_img = img_as_float64(io.imread(self.args.bf_path))
            bf_sum = np.sum(bf_img, axis=0) if bf_img.ndim == 3 else bf_img
            
            is_split_view = width > 1500
            bf_width_split = width // 2 if is_split_view else width
            
            if is_split_view:
                L_bf_full = bf_sum[:, :bf_width_split]
                R_bf_full = bf_sum[:, bf_width_split:]
                
                # Create the registered overlay
                if self.args.channel == "L":
                    # L is the static reference, R is the moving channel that gets transformed
                    R_reg = sr.transform(R_bf_full, tmats) if tmats is not None else R_bf_full
                    
                    # Replace StackReg's 0-padding with the mean background to prevent black lines
                    if tmats is not None:
                        R_reg[R_reg == 0] = np.mean(R_bf_full)
                        
                    bf_overlay_full = (L_bf_full + R_reg) / 2
                else:
                    # R is the static reference, L is the moving channel that gets transformed
                    L_reg = sr.transform(L_bf_full, tmats) if tmats is not None else L_bf_full
                    
                    # Replace StackReg's 0-padding with the mean background to prevent black lines
                    if tmats is not None:
                        L_reg[L_reg == 0] = np.mean(L_bf_full)
                        
                    bf_overlay_full = (R_bf_full + L_reg) / 2
                    
                # --- Save Brightfield Registration Overlay Plot ---
                if tmats is not None:
                    L_viz = cv2.normalize(L_bf_full, None, 0, 1, cv2.NORM_MINMAX)
                    R_viz = cv2.normalize(R_bf_full, None, 0, 1, cv2.NORM_MINMAX)
                    overlay_viz = cv2.normalize(bf_overlay_full, None, 0, 1, cv2.NORM_MINMAX)
                    
                    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
                    
                    # Panel 1: Original Left & Right unaligned overlay
                    axs[0].imshow((L_viz + R_viz) / 2, cmap='gray')
                    axs[0].set_title('Before: Original BF Images (Unaligned Overlay)')
                    axs[0].set_xticks([]); axs[0].set_yticks([])

                    # Panel 2: The transformed moving channel
                    if self.args.channel == "L":
                        axs[1].imshow(cv2.normalize(R_reg, None, 0, 1, cv2.NORM_MINMAX), cmap='gray')
                        axs[1].set_title('Registered Moving Channel (Right mapped to Left)')
                    else:
                        axs[1].imshow(cv2.normalize(L_reg, None, 0, 1, cv2.NORM_MINMAX), cmap='gray')
                        axs[1].set_title('Registered Moving Channel (Left mapped to Right)')
                    axs[1].set_xticks([]); axs[1].set_yticks([])

                    # Panel 3: After overlay
                    axs[2].imshow(overlay_viz, cmap='gray')
                    axs[2].set_title('After: Final Registered BF Overlay')
                    axs[2].set_xticks([]); axs[2].set_yticks([])
                    
                    plt.savefig(f"{self.args.save_dir}/BF_Registered_overlay.png", bbox_inches='tight')
                    plt.close()
                # -----------------------------------------------------
            else:
                bf_overlay_full = bf_sum

            # 4. Get the exact same ROI coordinates we used for the FL video
            y1, y2, x1, x2 = ImageProcessor.read_roi(
                self.args.roi_file, self.args.roi_channel, self.args.channel, 
                raw_shape[1], raw_shape[2], is_split_view
            )
            
            # 5. CROP the properly registered Brightfield overlay identically to the FL video
            bf_cropped = bf_overlay_full[y1:y2, x1:x2]
            
            # Normalize for saving
            bf_norm = cv2.normalize(bf_cropped, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

            # Save exactly as 'BF.tif' so fields_of_view.py can find it!
            tifffile.imwrite(f"{self.args.save_dir}/BF.tif", bf_norm, imagej=True)
            self.save_image_with_scalebar(bf_norm, f"{self.args.save_dir}/BF.png", cmap="gray")
            
            # Save the full uncropped reg image just in case you need it later
            full_bf_norm = img_as_uint(cv2.normalize(bf_overlay_full, None, 0, 1, cv2.NORM_MINMAX))
            tifffile.imwrite(f"{self.args.save_dir}/BF_reg_full.tif", full_bf_norm, imagej=True)

        return bf_sum, bf_cropped
    
    def save_image_with_scalebar(self, image, save_path, cmap="gray", bar_length_um=1):
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=cmap)
        ax.axis('off')
        
        # Calculate how many pixels make up the desired micrometer length
        pixels = bar_length_um / self.pxsize 
        
        scalebar = AnchoredSizeBar(
            ax.transData, 
            pixels, 
            f'{bar_length_um} µm', 
            'lower right', 
            pad=0.5, 
            color='white', 
            frameon=False, 
            size_vertical=max(1, int(image.shape[0] * 0.01)), # Automatically scales bar thickness
            fontproperties=fm.FontProperties(size=12)
        )
        ax.add_artist(scalebar)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def fit_cell(self, cell_mask, obj_num, channel, save_dir):
        contours = measure.find_contours(cell_mask, 0.5)
        if not contours: return None, None, None
        contour = max(contours, key=len)
        y_data, x_data = contour.T
        
        initial_R = (np.max(y_data) - np.min(y_data)) / 2.0
        initial_L = max(0.1, (np.max(x_data) - np.min(x_data)) - 2.0 * initial_R)
        initial_xc, initial_yc = np.mean(x_data), np.mean(y_data)
        
        def residuals_rotated(params, x_data, y_data):
            R, L, xc, yc, theta = params
            if R <= 0 or L < 0: return 1e9 * np.ones_like(x_data)
            x_rot = (x_data - xc) * np.cos(-theta) - (y_data - yc) * np.sin(-theta) + xc
            y_rot = (x_data - xc) * np.sin(-theta) + (y_data - yc) * np.cos(-theta) + yc
            
            x_start, x_end = -L / 2, L / 2
            y_model = np.full_like(x_rot, np.nan, dtype=float)
            x_shifted = x_rot - xc
            is_straight = (x_shifted >= x_start) & (x_shifted <= x_end)
            y_model[is_straight] = R
            is_left = (x_shifted < x_start)
            y_model[is_left] = np.sqrt(np.maximum(0, R**2 - (x_shifted[is_left] - x_start)**2))
            is_right = (x_shifted > x_end)
            y_model[is_right] = np.sqrt(np.maximum(0, R**2 - (x_shifted[is_right] - x_end)**2))
            
            res_top = (y_rot - yc) - y_model
            res_bottom = (y_rot - yc) - (-y_model)
            return np.where(np.abs(res_top) < np.abs(res_bottom), res_top, res_bottom)

        res = optimize.least_squares(
            residuals_rotated, [initial_R, initial_L, initial_xc, initial_yc, 0.0], 
            args=(x_data, y_data), bounds=([0.001, 0, -np.inf, -np.inf, -np.pi], [np.inf, np.inf, np.inf, np.inf, np.pi])
        )
        R_fit, L_fit, xc_fit, yc_fit, theta_fit = res.x
        fit_error = np.sqrt(np.sum(res.fun**2)) / len(x_data)
        
        # --- UNIT CORRECTION BUGFIX ---
        # Auto-detects if pxsize is in meters (51e-9) or microns (0.051)
        scale_factor = 1e9 if self.pxsize < 1e-4 else 1000
        
        cell_length = (2*R_fit + L_fit) * self.pxsize * scale_factor
        cell_width = 2*R_fit * self.pxsize * scale_factor
        error_nm = fit_error * self.pxsize * scale_factor

        print(f"\n--- Optimal Fit for cell {obj_num} ---", flush=True)
        print(rf"Radius (R): {R_fit:.2f} px, Length (L): {L_fit:.2f} px, Angle: {np.rad2deg(theta_fit):.1f}°", flush=True)
        print(f"Residual Error: {fit_error:.3f} px", flush=True)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cell_mask, cmap='binary', alpha=0.5)
        plt.plot(x_data, y_data, 'b.', markersize=2, alpha=0.5, label='Mask Contour')

        x_model_unrot = np.linspace(-L_fit/2 - R_fit, L_fit/2 + R_fit, 500)
        
        def unrot_y(x, R, L, top):
            y = np.full_like(x, np.nan, float)
            y[(x >= -L/2) & (x <= L/2)] = R if top else -R
            y[x < -L/2] = np.sqrt(np.maximum(0, R**2 - (x[x < -L/2] - (-L/2))**2)) * (1 if top else -1)
            y[x > L/2] = np.sqrt(np.maximum(0, R**2 - (x[x > L/2] - L/2)**2)) * (1 if top else -1)
            return y
            
        y_top = unrot_y(x_model_unrot, R_fit, L_fit, True)
        y_bot = unrot_y(x_model_unrot, R_fit, L_fit, False)
        
        x_m = np.concatenate([x_model_unrot, x_model_unrot[::-1]])
        y_m = np.concatenate([y_top, y_bot[::-1]])

        x_fitted = xc_fit + x_m * np.cos(theta_fit) - y_m * np.sin(theta_fit)
        y_fitted = yc_fit + x_m * np.sin(theta_fit) + y_m * np.cos(theta_fit)

        plt.plot(x_fitted, y_fitted, 'r-', linewidth=2, label='Fitted Outline')
        
        # --- RESTORED TITLE PARAMETERS ---
        title_text = (
            f'Fitted Cell Outline {obj_num}\n'
            rf'Length={cell_length:.2f} nm, Width={cell_width:.2f} nm, '
            rf'Center=({xc_fit:.2f}, {yc_fit:.2f}), $\theta$={np.rad2deg(theta_fit):.1f}° | '
            f'Error={error_nm:.3f} nm'
        )
        plt.title(title_text)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.savefig(f"{save_dir}/cell_mask_{obj_num}_{channel}_fitted.png")
        plt.close()

        # RETURN PARAMETERS FOR CSV SAVING & COORDS FOR HTML
        fit_params = {
            'Cell_Number': obj_num,
            'Channel': channel,
            'Length_nm': round(cell_length, 2),
            'Width_nm': round(cell_width, 2),
            'Center_X': round(xc_fit, 2),
            'Center_Y': round(yc_fit, 2),
            'Rotation_deg': round(np.rad2deg(theta_fit), 2),
            'Error_nm': round(error_nm, 3)
        }
        return fit_params, x_fitted, y_fitted
    
    
    def create_interactive_html(self, img, objects, fit_results_list, fit_outlines_dict, save_dir):
        print("Generating interactive HTML overlay with fits...", flush=True)
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fig = px.imshow(img_norm, color_continuous_scale='gray')
        area_thresh = int(self.args.area_filter)
        
        # Create a fast lookup dictionary for the fit parameters
        fit_dict = {f['Cell_Number']: f for f in fit_results_list}

        for obj in objects:
            status = "Accepted" if obj.area > area_thresh else "Rejected"
            color = None if status == "Accepted" else "rgba(255, 0, 0, 0.4)" 
            obj_num = obj.label

            if obj.image.shape[0] < 2 or obj.image.shape[1] < 2:
                continue

            # Grab fit data if it exists for this cell
            fit_data = fit_dict.get(obj_num)
            fit_outline = fit_outlines_dict.get(obj_num)

            # Build the text box
            hover_text = (
                f"<b>Object {obj_num}</b><br>"
                f"Status: {status}<br>"
                f"Area: {obj.area} px<br>"
            )
            if fit_data:
                hover_text += (
                    f"Length: {fit_data['Length_nm']} nm<br>"
                    f"Width: {fit_data['Width_nm']} nm<br>"
                    f"Error: {fit_data['Error_nm']} nm"
                )

            # Draw the real mask
            min_y, min_x, max_y, max_x = obj.bbox
            padded_image = np.pad(obj.image, pad_width=1, mode='constant', constant_values=0)
            contours = measure.find_contours(padded_image, 0.5)

            for contour in contours:
                global_y = contour[:, 0] + min_y - 1
                global_x = contour[:, 1] + min_x - 1

                fig.add_trace(go.Scatter(
                    x=global_x, y=global_y, mode='lines', fill='toself', fillcolor=color,
                    line=dict(color=color if color else None, width=1.5),
                    name=f"Cell {obj_num}", text=hover_text, hoverinfo="text", showlegend=False
                ))

            # Draw the mathematical fit as a dashed cyan line
            if fit_outline and status == "Accepted":
                x_fit, y_fit = fit_outline
                global_fit_x = x_fit
                global_fit_y = y_fit
                
                fig.add_trace(go.Scatter(
                    x=global_fit_x, y=global_fit_y, mode='lines',
                    line=dict(color='cyan', width=2, dash='dot'),
                    name=f"Fit {obj_num}", hoverinfo='skip', showlegend=False
                ))

        fig.update_layout(
            title=f"Interactive Cell Mask Overlay | Area Threshold: {area_thresh}",
            margin=dict(l=0, r=0, b=0, t=40), coloraxis_showscale=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False)
        )
        html_path = f"{save_dir}/interactive_cell_overlay.html"
        fig.write_html(html_path)
        print(f"Saved interactive overlay to: {html_path}", flush=True)
    
    
    def run(self):
        print(f"Starting analysis with mask type: {self.args.mask_type}", flush=True)
        
        # FIX: Try/except block added back around shutil.rmtree to prevent silent wrapper crashes
        if str(self.args.overwrite).lower() in ['true', '1', 't', 'y']:
            if "results" in self.args.save_dir and os.path.exists(self.args.save_dir):
                try:
                    shutil.rmtree(self.args.save_dir)
                except Exception as e:
                    print(f"Notice: Could not completely clear save directory (file might be open): {e}", flush=True)
                    pass 
                
        os.makedirs(self.args.save_dir, exist_ok=True)
            
        FileHandler.save_parameters(vars(self.args), self.args.save_dir)
        
        if hasattr(self.args, 'bead_path') and self.args.bead_path:
            target_tmat_path = "transformation_matrix.npy"
            if not os.path.exists(target_tmat_path):
                registrar = BeadRegistrar()
                if str(self.args.manual_registration).lower() in ['true', '1', 't', 'y']:
                    # Pass the single calibration image path
                    registrar.interactive_manual_registration(
                        self.args.bead_path, 
                        ".", 
                        int(self.args.manual_pairs)
                    )
                else:
                    # Run the existing automated StackReg logic
                    registrar.calculate_transformation(self.args.bead_path, ".", self.args.channel)
            self.args.tmats_path = target_tmat_path
            
        # 1. Video Processing
        L_chan, R_chan, num_frames, raw_shape = VideoProcessor.process_video(
            self.args.video_path, 0, int(self.args.num_frames), self.args.roi_channel, 
            self.args.roi_file, self.args.save_dir, int(self.args.num_channels), 
            self.args.channel, self.args.ALEX
        )
        
        if L_chan is not None:
            L_avg = np.mean(L_chan[:int(self.args.frame_avg)], axis=0).astype(np.uint16)
            tifffile.imwrite(f"{self.args.save_dir}/L_avg.tif", L_avg, imagej=True)
            self.save_image_with_scalebar(L_avg, f"{self.args.save_dir}/L_avg.png")
            tifffile.imwrite(f"{self.args.save_dir}/L_channel.tif", L_chan, imagej=True)
        if R_chan is not None:
            R_avg = np.mean(R_chan[:int(self.args.frame_avg)], axis=0).astype(np.uint16)
            tifffile.imwrite(f"{self.args.save_dir}/R_avg.tif", R_avg, imagej=True)
            self.save_image_with_scalebar(R_avg, f"{self.args.save_dir}/R_avg.png")
            tifffile.imwrite(f"{self.args.save_dir}/R_channel.tif", R_chan, imagej=True)

        width = raw_shape[2]
        img_for_masking = None
        sr = StackReg(StackReg.AFFINE)
        tmats = np.load(self.args.tmats_path) if self.args.tmats_path and os.path.exists(self.args.tmats_path) else None

        # 2. Frame Averaging & Brightfield Processing
        img_for_masking = None
        bf_sum = None 
        
        bf_sum, bf_cropped = self.process_brightfield(raw_shape, sr, tmats)
        
        # Set the image to be used for masking
        if self.args.mask_type == "BF" and self.args.bf_path:
            img_for_masking = bf_cropped 
        elif self.args.mask_type in ["FL_AI", "THRESHOLD", "AI", "WHOLE"]:
            target_chan = L_chan if self.args.channel == "L" else R_chan
            img_for_masking = np.mean(target_chan[:int(self.args.frame_avg)], axis=0).astype(np.uint16)
        
        # 3. Mask Generation & Labeling
        labels = None
        
        if self.args.mask_type in ["AI", "BF", "FL_AI"]:
            
            # --- PATH 1: OMNITORCH ---
            if self.args.model_type == "omnitorch":
                print("Omnipose execution deferred to external RunOmnipose script.", flush=True)
                import RunOmnipose
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                
                # 1. Generate the 3-channel visual mask via Omnipose
                visual_mask = RunOmnipose.main(img_for_masking, self.args.model, self.args.save_dir)
                io.imsave(f'{self.args.save_dir}/visual_mask_omni.tif', visual_mask, check_contrast=False)
                
                # 2. Create a blank 2D canvas based on the original image dimensions
                prediction = np.zeros(img_for_masking.shape, dtype=np.uint8)
                
                # 3. Find unique colors (cells) in the mask
                flat_pixels = visual_mask.reshape(-1, visual_mask.shape[-1])
                unique_colors = np.unique(flat_pixels, axis=0)
                unique_colors = [c for c in unique_colors if np.any(c > 0)] # Exclude black background
                
                print(f"Found {len(unique_colors)} unique objects.", flush=True)
                
                # 4. Loop through every unique cell color
                for i, color in enumerate(unique_colors):
                    
                    # Create a boolean mask where this specific color exists
                    mask_boolean = np.all(visual_mask == color, axis=-1)
                    
                    # Convert to an 8-bit image (255 for the cell, 0 for background)
                    object_mask = mask_boolean.astype(np.uint8) * 255
                    
                    # Save the individual cell mask (Optional, can be commented out to save disk space)
                    color_string = f"{color[0]}-{color[1]}-{color[2]}"
                    save_path = f'{self.args.save_dir}/object_mask_{color_string}.tif'
                    io.imsave(save_path, object_mask, check_contrast=False)
                    
                    # Erode the object and add it to our blank canvas
                    eroded_object = cv2.erode(object_mask, kernel=kernel, iterations=1)
                    prediction += eroded_object
                
                # 5. Cap the max value and output the final 1-channel mask
                prediction[prediction > 255] = 0
                print(f"Prediction shape {prediction.shape}", flush=True)    
                
                # Pass the processed 1-channel prediction to the rest of the pipeline
                mask = prediction
                
            # --- PATH 2: KERAS / U-NET ---
            elif self.args.model_type in ["unet", "keras"]:
                patches, h, w = ImageProcessor.make_patches(img_for_masking, self.args.inv_bf)
                
                thresh = 0.75
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                preds = []
                for i, p in enumerate(patches):
                    # 1. DEBUG: Show the input patch before neural net
                    plt.figure(figsize=(6, 5))
                    plt.imshow(p, cmap='gray')
                    plt.title(f"DEBUG: Input Patch {i+1} / {len(patches)}")
                    plt.colorbar()
                    plt.savefig(f"{self.args.save_dir}/patch.png")
                    plt.show() # Script will pause here until you close the plot window
                    
                    # Run the prediction
                    pred_patch = self.model.predict(p.reshape(1, p.shape[0], p.shape[1], 1))[0,:,:,0]
                    pred_patch[pred_patch < thresh] = 0
                    pred_patch[pred_patch >= thresh] = 1
                    
                    pred_patch = pred_patch * 255
                    preds.append(pred_patch)
                    
                    # 2. DEBUG: Show the neural net output
                    plt.figure(figsize=(6, 5))
                    plt.imshow(pred_patch, cmap='gray')
                    plt.title(f"DEBUG: Output Prediction {i+1} / {len(patches)}")
                    plt.colorbar()
                    plt.savefig(f"{self.args.save_dir}/predicted.png")
                    plt.show() # Script will pause here until you close the plot window
                
                #preds = [self.model.predict(p.reshape(1, p.shape[0], p.shape[1], 1))[0,:,:,0] * 255 for p in patches]
                mask = ImageProcessor.stitch_patches(preds, img_for_masking.shape, h, w)
                
            # --- PATH 3: STANDARD PYTORCH ---
            elif self.args.model_type == "pytorch":
                patches, h, w = ImageProcessor.make_patches(img_for_masking, self.args.inv_bf)
                preds = []
                with torch.no_grad():
                    for p in patches:
                        # Convert patch to tensor [Batch, Channel, Height, Width]
                        tensor_patch = torch.tensor(p, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        out = self.model(tensor_patch)
                        # Convert back to numpy array
                        pred_patch = out.squeeze().cpu().numpy() * 255
                        preds.append(pred_patch)
                mask = ImageProcessor.stitch_patches(preds, img_for_masking.shape, h, w)
                
        # --- FALLBACK: CLASSICAL THRESHOLDING ---
        elif self.args.mask_type == "THRESHOLD":
            mask = ImageProcessor.apply_watershed_threshold(
                img_for_masking, 
                dist_multiplier=0.3
            )
            
        elif self.args.mask_type == "WHOLE":
            mask = np.ones_like(img_for_masking, dtype=np.uint8) * 255
            
        # --- FALLBACK: LOAD FROM FILE ---
        elif self.args.mask_type == "file":
            mask = io.imread(glob.glob(f"{self.args.save_dir}/{self.args.mask_prefix}*.tif")[0])
            mask = (mask > 0).astype(np.uint8) * 255
            
        tifffile.imwrite(f"{self.args.save_dir}/all_cells_mask.tif", mask.astype(np.uint16), imagej=True)
        self.save_image_with_scalebar(mask, f"{self.args.save_dir}/all_cells_mask.png", cmap="viridis")

        # ==========================================================
        # Dual Mask Generation 
        # ==========================================================
    
        if int(self.args.num_channels) == 2 and self.args.channel == "L":
            mask_l, mask_r = ImageProcessor.generate_dual_masks(mask, self.args.channel)
        else:
            # If only 1 channel, strictly assign the mask to the target channel
            mask_l = mask if self.args.channel == 'L' else None
            mask_r = mask if self.args.channel == 'R' else None
        # ==========================================================

        # 4. Region Extraction
        # If we didn't use Omnipose, generate standard connected-component labels
        if labels is None:
            labels = measure.label(mask > 0)
            
        objects = measure.regionprops(labels)
        all_objects_data = []
        mask_stack_list = []
        all_fit_results = []
        fit_outlines = {}
        
        for obj_num, obj in enumerate(objects, start=1):
            if obj.area < int(self.args.area_filter): continue
            
            cell_dir = f"{self.args.save_dir}/cell_{obj_num}"
            os.makedirs(cell_dir, exist_ok=True)
            
            obj_data = {prop: getattr(obj, prop) for prop in ['label', 'area', 'bbox', 'centroid', 'eccentricity', 'major_axis_length', 'minor_axis_length', 'orientation', 'solidity', 'perimeter'] if hasattr(obj, prop)}
            all_objects_data.append(obj_data)
            
            cell_mask = (labels == obj.label).astype(np.uint8)
            mask_stack_list.append(cell_mask)
             
            extracted_intensities = {}    
            
            # Extract the bounding box for this specific cell
            min_y, min_x, max_y, max_x = obj.bbox
            mask_crop = cell_mask[min_y:max_y, min_x:max_x]

            for chan_name, chan_data in [("L", L_chan), ("R", R_chan)]:
                if chan_data is not None:
                    # 1. Extract mathematical intensities from the full video
                    intensities = []
                    for frame in chan_data[:num_frames]:
                        pixels = frame[cell_mask > 0]
                        intensities.append(np.mean(pixels) if len(pixels) > 0 else 0)
                        
                    extracted_intensities[chan_name] = intensities
                    
                    # 2. Create the ROI-sized 2D average image
                    # Average the first few frames of the already-ROI-cropped channel
                    chan_avg = np.mean(chan_data[:int(self.args.frame_avg)], axis=0).astype(np.uint16)
                    
                    # Create a copy to apply the mask
                    masked_roi_img = chan_avg.copy()
                    
                    # Identify the background (everything outside this specific cell's mask)
                    bg_mask = (cell_mask == 0)
                    
                    # Fill the background with the mean background value of those pixels
                    if np.any(bg_mask):
                        masked_roi_img[bg_mask] = np.mean(masked_roi_img[bg_mask])
                        
                    # Save the ROI-sized average as TIF
                    tifffile.imwrite(f"{cell_dir}/cellmasked_{chan_name}.tif", masked_roi_img, imagej=True)
                    
                    # Normalize to 8-bit and save as PNG for visual checking
                    cell_png_norm = cv2.normalize(masked_roi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    io.imsave(f"{cell_dir}/cellmasked_{chan_name}.png", cell_png_norm, check_contrast=False)
                    
                    # Plot Intensity and Fit Exponentials
                    frames_x = np.arange(len(intensities))
                    IntensityAnalyzer.plot_intensity([frames_x, intensities], chan_name, obj_num, len(intensities), cell_dir)
            
            mat_data = {
                'cell_num': obj_num,
                'area': obj.area,
                'centroid': obj.centroid,
                'bbox': obj.bbox,
                'cell_mask': cell_mask,
                'major_axis_length': obj.major_axis_length,
                'minor_axis_length': obj.minor_axis_length,
                'eccentricity': obj.eccentricity,
                'orientation': obj.orientation
            }
            
            if str(self.args.cell_fitting).lower() == "true":
                fit_result, x_fit, y_fit = self.fit_cell(cell_mask, obj_num, self.args.channel, cell_dir)
                if fit_result:
                    all_fit_results.append(fit_result)
                    fit_outlines[obj_num] = (x_fit, y_fit)
                    mat_data['fit_length_nm'] = fit_result['Length_nm']
                    mat_data['fit_width_nm'] = fit_result['Width_nm']

            if 'L' in extracted_intensities: mat_data['intensity_L'] = extracted_intensities['L']
            if 'R' in extracted_intensities: mat_data['intensity_R'] = extracted_intensities['R']
                
            sio.savemat(f"{cell_dir}/cell_{obj_num}_data.mat", mat_data)

        # 5. Save Summary Outputs
        if all_fit_results:
            df_fits = pd.DataFrame(all_fit_results)
            csv_path = f"{self.args.save_dir}/cell_fitting_parameters.csv"
            df_fits.to_csv(csv_path, index=False)
            print(f"\nSaved all cell fitting parameters to: {csv_path}", flush=True)
        if img_for_masking is not None:
            self.create_interactive_html(img_for_masking, objects, all_fit_results, fit_outlines, self.args.save_dir)
        if all_objects_data and mask_stack_list:
            stack_masks = np.array(mask_stack_list)
            tifffile.imwrite(f"{self.args.save_dir}/mask_stack.tif", stack_masks * 255, imagej=True)

            cleaned_objects_data = []
            for obj_dict in all_objects_data:
                row_data = {}
                for key, value in obj_dict.items():
                    if isinstance(value, (tuple, list, np.ndarray)): row_data[key] = str(value)
                    else: row_data[key] = value
                cleaned_objects_data.append(row_data)
            
            all_objects_df = pd.DataFrame(cleaned_objects_data)
            all_objects_df.to_csv(f"{self.args.save_dir}/all_objects.csv", index=False)
            
            is_split_view = raw_shape[2] > 1500
            y1, y2, x1, x2 = ImageProcessor.read_roi(
                self.args.roi_file, self.args.roi_channel, self.args.channel, 
                raw_shape[1], raw_shape[2], is_split_view
            )
            
            p = {
                'laser_centre_X': (x1 + x2) / 2, 
                'laser_centre_Y': (y1 + y2) / 2, 
                'laser_radius_R': (x2 - x1) / 2,
                'laser_angle': 0
            }
            
            matlab_dict = {
                'area': all_objects_df['area'].values,
                'BF': bf_sum if bf_sum is not None else np.zeros((raw_shape[1], raw_shape[2]//2)), 
                'bf2fl_tform': [0,0],
                'CellObject': stack_masks,  
                'p': p
            }
            
            if self.args.channel == "L" and L_chan is not None: matlab_dict['Cellframe0'] = L_chan[0]
            elif self.args.channel == "R" and R_chan is not None: matlab_dict['Cellframe0'] = R_chan[0]
            
            filename = self.args.save_dir + "/" + self.args.video_path.split("/")[-1][:-4] + "_segmentation.mat"
            sio.savemat(filename, matlab_dict)
            print(f"Saved global MATLAB segmentation data to {filename}", flush=True)

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].hist(all_objects_df['area'].dropna())
            ax[0].set_xlabel("Area (pixels)")
            ax[1].hist(all_objects_df['eccentricity'].dropna())
            ax[1].set_xlabel("Eccentricity")
            ax[2].scatter(all_objects_df['area'], all_objects_df['eccentricity'])
            ax[2].set_xlabel("Area")
            ax[2].set_ylabel("Eccentricity")
            plt.savefig(f"{self.args.save_dir}/obj_feature_summary.png")
            plt.close()

        print("Analysis pipeline completed successfully.", flush=True)

def run_preprocessing(params):
    """
    This is the entry point called by PySTACHIO.
    """
    print("\n" + "="*50, flush=True)
    print(f"PREPROCESSING SESSION: {params.name}", flush=True)
    print("="*50, flush=True)

    # Note: PySTACHIO has its own way of printing parameters, 
    # but we can keep a quick summary here for the preprocessing class
    print(f"Video Path  : {params.video_path}")
    print(f"Model       : {params.model}")
    print(f"ALEX Mode   : {params.ALEX}")
    print("="*50 + "\n", flush=True)

    try:
        # We pass the PySTACHIO params object directly into your class
        pipeline = AnalysisPipeline(params)
        pipeline.run()
        
    except Exception as e:
        # We keep your traceback logic so you still get detailed error reports
        print(f"\nCRITICAL ERROR DURING PREPROCESSING:\n{traceback.format_exc()}", flush=True)
        raise e # Re-raise so PySTACHIO knows the task failed