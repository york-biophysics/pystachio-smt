#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

"""

"""
import numpy as np

class Spots:
    def __init__(self, num_spots):
        self.num_spots = num_spots
        self.positions = np.zeros([num_spots, 2])
        self.clipping = [False] * num_spots
        self.bg_intensity = np.zeros(num_spots) 
        self.spot_intensity =  np.zeros(num_spots)
        self.centre_intensity =  np.zeros(num_spots)
        self.width = np.zeros([num_spots,2])
        self.frame = 0
        self.traj_num = np.array(range(num_spots))
        self.snr = np.zeros([num_spots,2])
        self.laser_on_frame = 0


#EJH#     def link():
#EJH#         return False
#EJH# 
#EJH#     def find_centre():
#EJH#         return False
#EJH# 
#EJH#     def distance_from(self, candidate):
#EJH#         dx = self.pos[x] - candidate.pos[x]
#EJH#         dy = self.pos[y] - candidate.pos[y]
#EJH#         return sqrt(dx**2 + dy**2)
#EJH# 
#EJH# def find_spots():
#EJH#     return False
#EJH# 
#EJH# def merge_coincedent_spots(candidate_spots, min_distance):
#EJH#     good_spots = []
#EJH#     for candidate in candidate_spots
#EJH#         is_good_spot = True
#EJH#         for good_spot in good_spots
#EJH#             if good_spot.distance_from(candidate) < min_distance:
#EJH#             is_good_spot = False
#EJH#             break
#EJH#         if is_good_spot:
#EJH#             good_spots.append(candidate)
#EJH# 
#EJH#     return good_spots