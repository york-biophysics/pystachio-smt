#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

"""

"""
import dash_html_components as html
import dash_core_components as dcc
from dash import dash_table
from dash.dependencies import Input,Output
import pandas as pd
import plotly.express as px


from dash_ui.app import app
import dash_ui.tabs.session
from trajectories import read_trajectories

empty_graph = {
    "layout": {
        "xaxis": {
            "visible": False
        },
        "yaxis": {
            "visible": False
        },
        "annotations": [
            {
                "text": "No matching data found",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 28
                }
            }
        ]
    }
}

def layout():
    info_pane = html.Div(id="info-pane",
        children=[
            html.H1("Information"),
            html.Div(id='info-content', children=[
                html.Div(id='info-stats', children=[
                    html.Table(id='info-stats-table', children=[
                        html.Tr([html.Td('Frame number:'),           html.Td("-", id='info-stats-frame-num')]),
                        html.Tr([html.Td('Number of Spots:'),        html.Td("-", id='info-stats-num-spots')]),
                        html.Tr([html.Td('Average Peak Intensity:'), html.Td("-", id='info-stats-avg-intensity')]),
                        html.Tr([html.Td('Average SNR:'),            html.Td("-", id='info-stats-avg-snr')]),
                        html.Tr([html.Td('Average Stoichiometry:'),  html.Td("-", id='info-stats-avg-stoich')]),
                ]),
                html.Div(id='info-graphs', children=[
                    dcc.Graph(id='info-stats-graph-isingle')

                ]),
            ])
        ])
    ])

    return info_pane

@app.callback([
        Output('info-stats-frame-num', 'children'),
        Output('info-stats-num-spots', 'children'),
        Output('info-stats-avg-intensity', 'children'),
        Output('info-stats-avg-snr', 'children'),
        Output('info-stats-avg-stoich', 'children'),
        Output('info-stats-graph-isingle', 'figure'),
    ], [
        Input('session-active-info-file-store', 'data'),
        Input('image-slider', 'value'),
    ])
def update_info(filename, frame_num):
    trajs = read_trajectories(filename)
    if not trajs: return ["-", "-", "-", "-", "-", empty_graph]
    num_spots = 0
    intensities = []
    avg_intensity = 0.0
    avg_snr = 0.0
    avg_snr = 0.0
    avg_stoich = 0.0
    for traj in trajs:
        intensities.extend([x for x in  traj.intensity])
        if frame_num >= traj.start_frame and frame_num <= traj.end_frame:
            num_spots += 1
            avg_intensity += traj.intensity[frame_num - traj.start_frame]
            avg_snr     += traj.snr[frame_num - traj.start_frame]
            avg_stoich  += traj.stoichiometry

    if num_spots > 0:
        avg_intensity /= num_spots
        avg_snr     /= num_spots
        avg_snr     /= num_spots
        avg_stoich  /= num_spots
    else:
        avg_intensity = '-'
        avg_snr     = '-'
        avg_snr     = '-'
        avg_stoich  = '-'

    intensity_fig = px.histogram(
            x = intensities,
            title='Peak Spot Intensities',
            labels = {'x':'Spot Intensity (camera counts/pixel)'},
            nbins=50,
    )

    return [frame_num, num_spots, avg_intensity, avg_snr, avg_stoich, intensity_fig]
