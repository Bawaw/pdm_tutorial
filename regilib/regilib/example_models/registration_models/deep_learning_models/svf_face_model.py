import torch
import numpy as np
import torch.nn as nn
import torch.distributions.normal as dist

from torch.nn import functional as F
from torch_geometric.nn import PointConv, fps, global_max_pool, radius
from regilib.registration_models.deep_learning_models.layers import create_identity, Exponentiation, GaussianSmoothing, PointcloudWarp

class FacePointModel(nn.Module):
    class SamplingGroupingConv(nn.Module):
        def __init__(self, in_channels, inter_channels, out_channels, ratio = 0.5, r = 0.2):
            """
            PointNet++ building block,

            input args:
               in_channels shape: [-1, in_channels + num_dimensions]
               out_channels shape: [-1, out_channels]
            """
            super(FacePointModel.SamplingGroupingConv, self).__init__()

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.ratio = ratio
            self.r = r

            self.conv = PointConv(
                nn.Sequential(
                    nn.Sequential(
                        nn.Linear(in_channels, inter_channels),
                        nn.ReLU(),
                        nn.BatchNorm1d(inter_channels)
                    ),
                    nn.Sequential(
                        nn.Linear(inter_channels, inter_channels),
                        nn.ReLU(),
                        nn.BatchNorm1d(inter_channels)
                    ),
                    nn.Sequential(
                        nn.Linear(inter_channels, out_channels),
                        nn.ReLU(),
                        nn.BatchNorm1d(out_channels)
                    )
                )
            )


        def forward(self, x, pos, batch):
            # Sampling Layer
            idx = fps(pos, batch, ratio=self.ratio)
            # Grouping Layer
            row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                              max_num_neighbors=64)
            edge_index = torch.stack([col, row], dim=0)

            # PointNet Layer
            x = self.conv(x, (pos, pos[idx]), edge_index)
            pos, batch = pos[idx], batch[idx]
            return x, pos, batch

    class GlobalPooling(nn.Module):
        def __init__(self, in_channels, inter_channels, inter_channels_2, out_channels):
            super(FacePointModel.GlobalPooling, self).__init__()

            self.nn = nn.Sequential(
                nn.Sequential(
                    nn.Linear(in_channels, inter_channels),
                    nn.ReLU(),
                    nn.BatchNorm1d(inter_channels)
                ),
                nn.Sequential(
                    nn.Linear(inter_channels, inter_channels_2),
                    nn.ReLU(),
                    nn.BatchNorm1d(inter_channels_2)
                ),
                nn.Sequential(
                    nn.Linear(inter_channels_2, out_channels),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels)
                )
            )

        def forward(self, x, pos, batch):
            x = self.nn(torch.cat([x, pos], dim=1))
            x = global_max_pool(x, batch)
            return x

    def __init__(self, bounds, grid_size, d = 2, feature_channels = 0, zdim = 2, n_samples = 1,
                 kernel_size = 15, sigma_sq = 2**1):
        super(FacePointModel, self).__init__()

        self.zdim, self.n_samples, self.d = zdim, n_samples, d
        self.bounds, self.grid_size = bounds, grid_size

        self.identity, _ = create_identity(bounds, grid_size)

        # fixed and moving encoder
        self.conv1_1 = FacePointModel.SamplingGroupingConv(self.d + feature_channels, 64, 128, ratio = 0.5, r = 0.2)
        self.conv1_2 = FacePointModel.SamplingGroupingConv(128 + self.d, 128, 256, ratio = 0.25, r = 0.4)
        self.global_pool_1 = FacePointModel.GlobalPooling(256 + self.d, 256, 512, 1024)

        # conv with kernel_size 1 to merge in feature dimension
        #self.conv4 = nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = 1)

        ## mu
        self.fc11 = nn.Linear(in_features=1024, out_features=512)
        self.fc12 = nn.Linear(in_features=512, out_features=self.zdim)

        ## log_variance
        self.fc21 = nn.Linear(in_features=1024, out_features=512)
        self.fc22 = nn.Linear(in_features=512, out_features=self.zdim)

        # decoder
        transpose_convolution = nn.ConvTranspose3d if self.d == 3 else nn.ConvTranspose2d
        batch_norm = nn.BatchNorm3d if self.d == 3 else nn.BatchNorm2d
        convolution = nn.Conv3d if self.d == 3 else nn.Conv2d


        self.fc1 = nn.Linear(in_features=self.zdim, out_features=1024)

        self.fc2_out_features = ([16]*self.d)
        self.fc2 = nn.Linear(in_features=1024,
                             out_features= 4 * np.prod(self.fc2_out_features))

        self.conv_t1 = transpose_convolution(in_channels=4, out_channels=16,
                                             kernel_size=4, padding=1, stride=2)
        self.batchnorm_t1 = batch_norm(16)

        # channel here is the dimension of the velocity vector
        self.conv_t2 = transpose_convolution(in_channels=16, out_channels=33,
                                             kernel_size=4, padding=1, stride=2)
        self.batchnorm_t2 = batch_norm(33)
        self.conv5 = convolution(in_channels = 33, out_channels = 16, kernel_size = 1)
        self.batchnorm_5 = batch_norm(16)
        self.conv6 = convolution(in_channels = 16, out_channels = self.d, kernel_size = 1)

        self.exponent = Exponentiation(self.bounds)
        self.smooth = GaussianSmoothing(channels = self.d, dim = self.d,
                                        kernel_size = [kernel_size]*self.d, sigma_sq = sigma_sq)
        self.warp = PointcloudWarp(self.bounds)

    def to(self, device):
        self.identity = self.identity.to(device)
        return super().to(device)

    def encode(self, _fixed, _moving):
        fixed, moving = _fixed.clone(), _moving.clone()


        # import pyvista as pv
        # pv.set_plot_theme("document")
        # plot = pv.Plotter(shape=(1, 1))
        # pc = pv.PolyData(fixed.pos.detach().cpu().numpy())
        # plot.add_mesh(pc, eye_dome_lighting=True,render_points_as_spheres=True, point_size = 30., scalars=torch.zeros(fixed.pos.shape[0]).numpy())
        # plot.camera_position = [(3, 0, 4), (0, 0, 0), (0, 1, 0)]
        # plot.show()

        # encode fixed graph
        fixed.x, fixed.pos, fixed.batch = self.conv1_1(fixed.x, fixed.pos, fixed.batch)

        # pv.set_plot_theme("document")
        # plot = pv.Plotter(shape=(1, 1))
        # pc = pv.PolyData(fixed.pos.detach().cpu().numpy())
        # plot.add_mesh(pc, eye_dome_lighting=True,render_points_as_spheres=True, point_size = 30.,scalars=fixed.x.detach().cpu().mean(1).numpy())
        # plot.camera_position = [(3, 0, 4), (0, 0, 0), (0, 1, 0)]
        # plot.show()

        fixed.x, fixed.pos, fixed.batch = self.conv1_2(fixed.x, fixed.pos, fixed.batch)

        # pv.set_plot_theme("document")
        # plot = pv.Plotter(shape=(1, 1))
        # pc = pv.PolyData(fixed.pos.detach().cpu().numpy())
        # plot.add_mesh(pc, eye_dome_lighting=True,render_points_as_spheres=True, point_size = 30.,scalars=fixed.x.detach().cpu().mean(1).numpy())
        # plot.camera_position = [(3, 0, 4), (0, 0, 0), (0, 1, 0)]
        # plot.show()

        x_f = self.global_pool_1(fixed.x, fixed.pos, fixed.batch)

        x = x_f.view(-1, self.fc11.in_features)

        # mu normal parameter
        mu_z = F.elu(self.fc11(x))
        mu_z = self.fc12(mu_z)

        # log(var) parameter
        mu_logvar = F.elu(self.fc21(x))
        mu_logvar = self.fc22(mu_logvar)

        return mu_z, mu_logvar

    def decode(self, z, _conditional):
        conditional = _conditional.clone()
        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = x.view(-1, 4, *self.fc2_out_features)

        x = self.batchnorm_t1(F.elu(self.conv_t1(x)))

        x = self.batchnorm_t2(F.elu(self.conv_t2(x)))
        x = self.batchnorm_5(F.elu(self.conv5(x)))
        velocity_field = self.conv6(x)
        return self.exponentiate_and_warp(conditional, velocity_field)

    def exponentiate_and_warp(self, moving, velocity_field):
        smooth_velocity_field = self.smooth(velocity_field.clone())
        phi = self.exponent(self.identity, smooth_velocity_field)
        x = self.warp(moving, phi)

        return velocity_field, phi, x

    def sample_z(self, mu, logvar, take_mean = False):
        if take_mean:
            return mu

        std = torch.exp(0.5*logvar)
        samples = dist.Normal(mu,std).rsample((self.n_samples,))
        return samples

    def forward(self, fixed, moving, testing = False):
        # encode
        mu, logvar = self.encode(fixed, moving)

        # use this when using regular autoencoder
        # z = mu

        # # sample latent space
        # only take the first sample
        # comment this when using regular AE
        z = self.sample_z(mu, logvar)[0]#, testing)

        # unfold sample dimension and treat as part of the batch
        velocity_field, phi, x = self.decode(z, moving)

        return velocity_field, phi, x, mu, logvar

    def warp_attr(self, data_f = None, data_m = None, phi = None, attr = 'pos', testing = False):
        # if not batched we assume they form a single batch
        if (data_f is not None) and ('batch' not in data_f.keys):
            data_f.batch = torch.zeros(data_f.pos.shape[0], device = data_f.pos.device, dtype = torch.long)
        if (data_m is not None) and ('batch' not in data_m.keys):
            data_m.batch = torch.zeros(data_m.pos.shape[0], device = data_m.pos.device, dtype = torch.long)

        if phi is None and data_f is not None:
            _, phi, _, _, _ = self.forward(data_f, data_m, testing)

        warp = PointcloudWarp(self.bounds, attr)
        return warp(data_m, phi)
