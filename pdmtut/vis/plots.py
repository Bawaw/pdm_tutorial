#!/usr/bin/env python3

import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import seaborn as sns
import torch


def plot_representation(z_coordinates, **kwargs):
    if z_coordinates.shape[-1] == 2:
        _plot_representation_2d(z_coordinates, **kwargs)
    if z_coordinates.shape[-1] == 3:
        _plot_representation_3d(z_coordinates, **kwargs)

def _plot_representation_2d(
        z_coordinates, index_colors=None, z_extremes=None,
        interpolate_background=False, root=None, axis=None):

    data = {
        '$z_1$' : z_coordinates[:, 0],
        '$z_2$' : z_coordinates[:, 1]
    }

    # distribution plot
    g = sns.jointplot(
        data=data, x="$z_1$", y="$z_2$", zorder=100, s=80, edgecolor="#202020",
        joint_kws={'color':None, 'c':index_colors.tolist()},
        ax=axis
    )
    g.fig.set_figwidth(10); g.fig.set_figheight(10)
    g.ax_joint.set_xlabel('$z_1$', fontsize=15)
    g.ax_joint.set_ylabel('$z_2$', fontsize=15)

    # interpolate background
    if interpolate_background and index_colors is not None:
        from scipy.interpolate import NearestNDInterpolator

        z_min = np.floor(z_coordinates.min(0))
        z_max = np.ceil(z_coordinates.max(0))
        if z_extremes is not None:
            z_min = np.vstack([z_min, np.floor(z_extremes.numpy().min(0))]).min(0)
            z_max = np.vstack([z_max, np.ceil(z_extremes.numpy().max(0))]).max(0)

        z_range = (z_min, z_max)

        X, Y = np.meshgrid( # 2D grid for interpolation
            np.linspace(z_range[0][0], z_range[1][0], 10),
            np.linspace(z_range[0][1], z_range[1][1], 10),
        )

        interp = NearestNDInterpolator(z_coordinates, y=index_colors)
        Z = interp(X, Y)

        g.ax_joint.scatter(
            X.flatten(), Y.flatten(), c=Z.reshape(-1, 3),
            linewidth=0., marker='s', s=2000, alpha=0.5)

    # plot extreme points and trajectories
    if z_extremes is not None:
        n_sets = math.floor(z_extremes.shape[0] / 2)
        extreme_data = {
            '$z_1$': z_extremes[:, 0],
            '$z_2$': z_extremes[:, 1],
            'set': torch.cat([n*torch.ones(2) for n in range(n_sets)])
        }

        # plot extreme points
        sns.scatterplot(
            data=extreme_data, x='$z_1$', y='$z_2$', hue='set', legend=False,
            s=200, linewidth=2, ax=g.ax_joint, zorder=200, edgecolor="#404040",
            palette=[(1., 0., 0.), (0., 1, 0.), (0., 0, 1.)]
        )

        # plot line between extreme points 1
        sns.lineplot(
            data=extreme_data, x='$z_1$', y='$z_2$', hue='set',
            lw=2, ax=g.ax_joint, zorder=100, legend=False,
            palette=[(1., 0, 0.), (0., 1, 0.), (0., 0, 1.)]
        )

        if root is not None:
            plt.savefig(os.path.join(root, 'base_representation.png'))

            pickle.dump({
                'z_coordinates': z_coordinates,
                'index_colors': index_colors,
                'z_extremes': z_extremes,
                'interpolate_background': interpolate_background
            }, open(os.path.join(root, 'base_representation.obj'), 'wb'))

        if axis is None: plt.show()


def _plot_panel(data, extreme_data, k1, k2, index_colors, ax):
    sns.scatterplot(
        data=data, x=k1, y=k2, c=index_colors, ax=ax)

    if extreme_data is not None:

        # plot extreme points
        sns.scatterplot(
            data=extreme_data, x=k1, y=k2, hue='set', legend=False,
            s=200, linewidth=2, zorder=200, ax=ax,
            palette=[(1., 0., 0.), (0., 1, 0.), (0., 0, 1.)]
        )

        # plot line between extreme points 1
        sns.lineplot(
            data=extreme_data, x=k1, y=k2, hue='set',
            lw=2, zorder=100, legend=False, ax=ax,
            palette=[(1., 0., 0.), (0., 1, 0.), (0., 0, 1.)]
        )


def _plot_representation_3d(
        z_coordinates, index_colors=None, z_extremes=None,
        interpolate_background=False, root=None, axis=None):

    data = {
        '$z_1$': z_coordinates[:, 0],
        '$z_2$': z_coordinates[:, 1],
        '$z_3$': z_coordinates[:, 2]
    }

    if z_extremes is not None:
        n_sets = math.floor(z_extremes.shape[0] / 2)
        extreme_data = {
            '$z_1$': z_extremes[:, 0],
            '$z_2$': z_extremes[:, 1],
            '$z_3$': z_extremes[:, 2],
            'set': torch.cat([n*torch.ones(2) for n in range(n_sets)])
        }
    else:
        extreme_data = None

    fig, ax = plt.subplots(2, 2)
    _plot_panel(data, extreme_data, '$z_1$', '$z_2$', index_colors, ax[0,0])
    _plot_panel(data, extreme_data, '$z_3$', '$z_2$', index_colors, ax[0,1])
    _plot_panel(data, extreme_data, '$z_1$', '$z_3$', index_colors, ax[1,0])

    ax[1, 1].axis('off')
    if root is not None:
        plt.savefig(os.path.join(root, 'base_representation.png'))

        pickle.dump({
            'z_coordinates': z_coordinates,
            'index_colors': index_colors,
            'z_extremes': z_extremes,
            'interpolate_background': interpolate_background
        }, open(os.path.join(root, 'base_representation.obj'), 'wb'))

    if axis is None: plt.show()

def plot_reconstruction(
        reconstructed_state, index_colors, root=None, axis=None):
    plotter = pv.Plotter() if axis is None else axis

    plotter.add_mesh(
        pv.PolyData(reconstructed_state),
        render_points_as_spheres=True, point_size=10,
        diffuse=0.99, specular=0.8, ambient=0.3, smooth_shading=True,
        scalars=index_colors,
        style='points', rgb=True
    )

    if axis is None:
        plotter.camera_position = [(-65, 0, 65), (0, 0, 0), (0, 1, 0)]
        _ = plotter.show(window_size=[800, 800])

    if root is not None:
        plotter.screenshot(os.path.join(root, 'reconstruction.png'))
        pickle.dump({
            'reconstructed_state': reconstructed_state,
            'index_colors': index_colors
        }, open(os.path.join(root, 'reconstruction.obj'), 'wb'))


def plot_density(reconstructed_state, log_prob, root=None, axis=None):
    plotter = pv.Plotter() if axis is None else axis

    plotter.add_mesh(
        pv.PolyData(reconstructed_state),
        render_points_as_spheres=True, point_size=10,
        diffuse=0.99, specular=0.8, ambient=0.3, smooth_shading=True,
        scalars=log_prob,
        style='points', scalar_bar_args={'title':'Log probability'}
    )

    if axis is None:
        plotter.camera_position = [(-65, 0, 65), (0, 0, 0), (0, 1, 0)]
        _ = plotter.show(window_size=[800,800])

    if root is not None:
        plotter.screenshot(os.path.join(root, 'discrete_density.png'))
        pickle.dump({
            'reconstructed_state': reconstructed_state,
            'log_prob': log_prob
        }, open(os.path.join(root, 'discrete_density.obj'), 'wb'))

def plot_generated_samples(generated_samples, log_prob, root=None, axis=None):
    plotter = pv.Plotter() if axis is None else axis

    plotter.add_mesh(
        pv.PolyData(generated_samples),
        render_points_as_spheres=True, point_size=10,
        diffuse=0.99, specular=0.8, ambient=0.3, smooth_shading=True,
        scalars=log_prob,
        style='points', scalar_bar_args={'title':'Log probability'}
    )

    if axis is None:
        plotter.camera_position = [(-65, 0, 65), (0, 0, 0), (0, 1, 0)]
        _ = plotter.show(window_size=[800,800])

    if root is not None:
        plotter.screenshot(os.path.join(root, 'generated_samples.png'))
        pickle.dump({
            'generated_samples': generated_samples,
            'log_prob': log_prob
        }, open(os.path.join(root, 'generated_samples.obj'), 'wb'))

def plot_interpolation(
        interpolated_points_1, interpolated_points_2, interpolated_points_3,
        mesh, mesh_log_prob, root=None, axis=None):
    plotter = pv.Plotter() if axis is None else axis

    plotter.add_mesh(
        pv.StructuredGrid(*mesh),
        scalars=mesh_log_prob, style='surface', pbr=True, metallic=0.2, roughness=0.6,
        opacity=0.4, scalar_bar_args={'title':'Log probability'}
    )

    plotter.add_mesh(pv.Spline(interpolated_points_1, 100).tube(0.1), color=(1., 0., 0.))
    plotter.add_mesh(
        pv.PolyData(np.stack([interpolated_points_1[0], interpolated_points_1[-1]])),
        render_points_as_spheres=True, point_size=12,
        diffuse=0.99, specular=0.8, ambient=1., smooth_shading=True,
        style='points', rgb=True, color=(1., 0., 0.)
    )

    plotter.add_mesh(pv.Spline(
        interpolated_points_2, 100).tube(0.1), color=(0., 1, 0.))
    plotter.add_mesh(
        pv.PolyData(np.stack([interpolated_points_2[0], interpolated_points_2[-1]])),
        render_points_as_spheres=True, point_size=12,
        diffuse=0.99, specular=0.8, ambient=1., smooth_shading=True,
        style='points', rgb=True, color=(0., 1, 0.)
    )

    plotter.add_mesh(pv.Spline(
        interpolated_points_3, 100).tube(0.1), color=(0., 0, 1.))
    plotter.add_mesh(
        pv.PolyData(np.stack([interpolated_points_3[0], interpolated_points_3[-1]])),
        render_points_as_spheres=True, point_size=12,
        diffuse=0.99, specular=0.8, ambient=1., smooth_shading=True,
        style='points', rgb=True, color=(0., 0, 1.)
    )

    plotter.add_light(pv.Light(
        position=(-65, 0, -65), show_actor=True, positional=True,
        cone_angle=100, intensity=1.))
    plotter.add_light(pv.Light(
        position=(0, 0, -65), show_actor=True, positional=True,
        cone_angle=100, intensity=1.))

    if axis is None:
        plotter.camera_position = [(-65, 0, 65), (0, 0, 0), (0, 1, 0)]
        plotter.show(window_size=[800,800])

    if root is not None:
        plotter.screenshot(os.path.join(root, 'interpolation.png'))
        pickle.dump({
            'interpolated_points_1': interpolated_points_1,
            'interpolated_points_2': interpolated_points_2,
            'interpolated_points_3': interpolated_points_3,
            'mesh': mesh,
            'mesh_log_prob': mesh_log_prob
        }, open(os.path.join(root, 'interpolation.obj'), 'wb'))
