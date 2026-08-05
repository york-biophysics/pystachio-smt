#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

""" ALGORITHMS - Low level algorithms module

Description:
    algorithms.py contains a number of useful algorithms that are used
    throughout the code, but don't necessarily need any of the data structures
    defined in other modules.

Contains:
    function fwhm
    function get_distance_list
    function find_local_maxima
    function ultimate_erode
    function gaussian
    functino moments
    function fit_gaussian
    function rotate_3d_vector

Author:
    Edward Higgins & JWS

Version: 0.2.0
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from numba import jit

def fwhm(data):
    """FWHM - Calculates the width of the highest peak in data

    Description:
        fwhm calculates the full width at half maximum of the highest peak in
        `data`, a 1D numpy array. Both the position and width of the peak are
        returned relative to the array indices.

        If the peak goes over the edge of the array, the width is set to None.

    Inputs:
        np.array([N]): data
            Input data containing the peak

    Outputs:
        float: x_width
            Width of the peak

        float: extremum_val
            Index of the peak position
    """

    x = np.linspace(0, 255, 256).astype(int)

    data = data / np.max(data)
    N = data.size - 1

    lev50 = 0.5
    if data[0] < lev50:
        centre_index = np.argmax(data)
        Pol = +1
    else:
        centre_index = np.argmin(data)
        Pol = -1

    if centre_index > 254:
        print("WARNING: Struggling to find a peak in histogram")
        centre_index = 254

    extremum_val = x[centre_index]

    i = 1
    while np.sign(data[i] - lev50) == np.sign(data[i - 1] - lev50) and i < 255:
        i += 1

    interp = (lev50 - data[i - 1]) / (data[i] - data[i - 1])
    lead_t = x[i - 1] + interp * (x[i] - x[i - 1])

    i = centre_index + 1
    while (np.sign(data[i] - lev50) == np.sign(data[i - 1] - lev50)) and (i <= N - 1):
        i += 1

    if i != N:
        p_type = 1
        interp = (lev50 - data[i - 1]) / (data[i] - data[i - 1])
        trail_t = x[i - 1] + interp * (x[i] - x[i - 1])
        x_width = trail_t - lead_t
    else:
        p_type = 2
        trail_t = None
        x_width = 0

    return (x_width, extremum_val)


def get_distance_list(r_max):
    L = 2 * r_max + 1
    distance_map = np.zeros([L, L])
    for i in range(L):
        for j in range(L):
            distance_map[i, j] = np.sqrt((i - r_max) ** 2 + (j - r_max) ** 2)

    distance_list = []
    for i in range(L):
        for j in range(L):
            if distance_map[i, j] <= r_max:
                distance_list.append([i - r_max, j - r_max, distance_map[i, j]])

    distance_list.sort(key=lambda x: x[2])

    return distance_list


@jit(nopython=True)
def find_local_maxima(img):
    local_maxima = []
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if (
                img[i, j] != 0
                and np.max(img[i - 1 : i + 2, j - 1 : j + 2]) == img[i, j]
            ):
                local_maxima.append([j, i])

    return local_maxima


def ultimate_erode(img, orig):
    distance_list = np.array(get_distance_list(16))

    img_dist = uer_jittable(img, distance_list)

    spot_locations = find_local_maxima(img_dist)

    if not spot_locations:
        spot_locations = []

    return spot_locations

@jit(nopython=True)
def uer_jittable(img, distance_list):
    img_dist = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 0:
                for pixel in distance_list:
                    if (
                        i + pixel[0] < 0
                        or i + pixel[0] >= img.shape[0]
                        or j + pixel[1] < 0
                        or j + pixel[1] >= img.shape[1]
                    ):
                        img_dist[i, j] = pixel[2]
                        break
                        continue

                    if img[i + int(pixel[0]), j + int(pixel[1])] == 0:
                        img_dist[i, j] = pixel[2]
                        break
                if img_dist[i, j] == 0:
                    print(f"WARNING: Unable to find any spots in this frame")
                    return np.zeros(img.shape)
    return img_dist

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    # total = data.sum()
    # X, Y = np.indices(data.shape)
    # x = (X*data).sum()/total
    # y = (Y*data).sum()/total
    # if int(y)>=16 or int(y)<0: y=8
    # if int(x)>=16 or int(x)<0: x=8
    # col = data[:, int(y)]
    # width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    # row = data[int(x), :]
    # width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    # height = data.max()
    return 1000, 8, 8, 1.5, 1.5 #height, x, y, width_x, width_y

def fit2Dgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p, success

def rotate_3d_vector(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
