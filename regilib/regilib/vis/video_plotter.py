#!/usr/bin/env python3

import torch
import pyvista as pv
from regilib.vis.plotter import Plotter

def plot_video(time_sequence, file_name, camera_pos=None, reverse=False, scalars = None,
               theme='document', show_grid = False, trace=False, sticky_first=False,
               sticky_last=False, **kwargs):
    if isinstance(time_sequence, torch.Tensor):
        time_sequence = torch.unbind(time_sequence)
    if scalars is None:
        scalars = [None]*len(time_sequence)
    if scalars is not None and isinstance(scalars, torch.Tensor):
            scalars = torch.unbind(scalars)
    if trace:
        trajectories = [
            pv.Spline(points, 100).tube(0.02) for points in torch.stack(
                time_sequence).permute(1,0,2)]

    T = len(time_sequence)

    pv.set_plot_theme(theme)
    plotter = Plotter()

    if camera_pos is not None:
        plotter.camera_position = camera_pos

    if show_grid: plotter.show_grid()
    if trace:
        for traj in trajectories: plotter.add_mesh(traj, color='gray', opacity=0.7)
    if sticky_first: plotter.add_generic(
            time_sequence[0], scalars=scalars[0], opacity=0.7, **kwargs)
    if sticky_last: plotter.add_generic(
            time_sequence[-1], scalars=scalars[-1], opacity=0.7, **kwargs)

    main_shape = plotter.add_generic(
        time_sequence[0], scalars=scalars[0], **kwargs)

    plotter.show(auto_close=False)
    plotter.open_gif(file_name)

    for i in range(1, T):
        plotter.update_coordinates(time_sequence[i], mesh=main_shape)
        if scalars is not None and len(scalars) > 1:
            plotter.update_scalars(scalars[i])
        plotter.write_frame()

    if reverse:
        for i in range(T-1, 0, -1):
            plotter.update_coordinates(time_sequence[i], mesh=main_shape)
            if scalars is not None and len(scalars) > 1:
                plotter.update_scalars(scalars[i])
            plotter.write_frame()

    plotter.close()
