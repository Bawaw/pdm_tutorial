#!/usr/bin/env python3

import torch
import pyvista as pv
from regilib.vis.plotter import Plotter

def plot_trajectory(plotter, time_sequence, camera_pos=None, scalars = None,
               theme='document', show_grid = False, trace=False, sticky_first=False,
               sticky_last=False, start_point_color=None, end_point_color=None,
                    trajectory_color=None, **kwargs):

    if isinstance(time_sequence, torch.Tensor):
        time_sequence = torch.unbind(time_sequence)
    if scalars is None:
        scalars = [None]*len(time_sequence)
    if scalars is not None and isinstance(scalars, torch.Tensor):
            scalars = torch.unbind(scalars)
    if trace:
        trajectories = [pv.Spline(points, 100).tube(0.009) for points in torch.stack(time_sequence).permute(1,0,2)]

    T = len(time_sequence)

    pv.set_plot_theme(theme)

    if camera_pos is not None:
        plotter.camera_position = camera_pos

    if show_grid: plotter.show_grid()
    if trace:
        for traj in trajectories: plotter.add_mesh(traj, color=trajectory_color, opacity=0.7)
    if sticky_first: plotter.add_generic(
            time_sequence[0], scalars=scalars[0], opacity=1., color=start_point_color, **kwargs)
    if sticky_last: plotter.add_generic(
            time_sequence[-1], scalars=scalars[-1], opacity=1., color=end_point_color, **kwargs)


    return plotter
