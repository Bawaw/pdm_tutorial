import math
import torch
import regilib.registration_models.deep_learning_models.utils as utils
import torch.nn as nn
import torch.nn.functional as F

class Exponentiation(nn.Module):
    def __init__(self, bounds):
        super(Exponentiation, self).__init__()
        self.bounds = bounds

    def forward(self, identity, u, recorder = None):
        """
            Computes the diffeomorphic field parameterised by stationary velocity field u, phi = Exp(u)

            input args:
                identity: the identity transform x
                    shapes: (B, C, H_in, W_in)
                u: stationary velocity field u
                    shapes: (B, C, H_in, W_in)
            output:
                phi: diffeomorphic deformation field
                    shapes: (B, C, H_in, W_in)

        """

        # identity is a matrix function so that when it is applied to the image the same image will come out
        # it is thus just the indexing of the image
        N = torch.tensor(7.)

        phi_step = identity + (u/(2**N))
        if recorder is not None:
            recorder.append(phi_step)

        for i in range(N.int().item() - 1):
            # sample_grid expects xy coordinates so we remain in the xy coordinate system
            phi_step = compose_deformation(phi_step, phi_step, self.bounds)

            if recorder is not None:
                recorder.append(phi_step)


        return phi_step

class PointcloudWarp(nn.Module):
    def __init__(self, bounds, pos_key = 'pos'):
        super(PointcloudWarp, self).__init__()
        self.bounds = bounds
        self.pos_key = pos_key

    def forward(self, pointcloud, phi):
        # convert the concatenated graph vertices into a BxNxd tensor
        pc_out = pointcloud.clone()

        batch_key = 'batch' if self.pos_key == 'pos' else '{}_batch'.format(self.pos_key)
        assert pointcloud[batch_key].shape[0] == pointcloud[self.pos_key].shape[0], \
            "batch and vertices shape differ"

        sample_grid = utils.cat_tensor_to_batch_tensor(
            pc_out[self.pos_key], pointcloud[batch_key])

        dims = sample_grid.shape[-1]

        # sample grid with positions to sample
        sample_grid = (sample_grid[:,:,None,:]
                       if dims == 2
                       else sample_grid[:,:,None,None,:])


        # sample the displacement field for every coordinate in the pointcloud
        new_grid = compose_deformation(phi, sample_grid, self.bounds)

        new_grid = new_grid[...,0] if dims == 2 else new_grid[...,0,0]
        new_grid = new_grid.permute(0,2,1)
        new_grid = utils.batch_tensor_to_cat_tensor(new_grid)

        pc_out[self.pos_key] = new_grid

        return pc_out

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def G2(self,x,y):
        return ((1/(2*math.pi*self.sigma_sq)) *\
            torch.exp(-((x-self.mean[0])**2+(y-self.mean[1])**2)/(2*self.sigma_sq)))

    def G3(self,x,y,z):
        return ((1/(2*math.pi*self.sigma_sq)) *
                torch.exp(-((x-self.mean[0])**2+
                            (y-self.mean[1])**2+
                            (z-self.mean[2])**2) /(2*self.sigma_sq)
                ))


    def __init__(self, channels = 2, dim = 2, kernel_size = None, sigma_sq = 1.):
        super(GaussianSmoothing, self).__init__()
        if kernel_size == None:
            kernel_size = [5]*dim

        assert len(kernel_size) == dim

        self.sigma_sq = sigma_sq
        self.kernel_size = kernel_size
        self.channels = channels
        self.dim = dim

        self.mean = tuple(map(lambda x: (x-1) /2, self.kernel_size))

        if dim == 2:
            meshgrid = torch.meshgrid(
                    torch.arange(kernel_size[0], dtype=torch.float32),
                    torch.arange(kernel_size[1], dtype=torch.float32)
            )
            self.kernel = self.G2(*meshgrid)
            self.kernel /= torch.sum(self.kernel) # normalise to sum of 1
            # shape(out_channels, in_channels/groups, kH,kW)
            self.kernel = self.kernel.repeat((self.channels,1,1,1))
        if dim == 3:
            meshgrid = torch.meshgrid(
                    torch.arange(kernel_size[0], dtype=torch.float32),
                    torch.arange(kernel_size[1], dtype=torch.float32),
                    torch.arange(kernel_size[2], dtype=torch.float32)
            )
            self.kernel = self.G3(*meshgrid)
            self.kernel /= torch.sum(self.kernel) # normalise to sum of 1
            # shape(out_channels, in_channels/groups, kT,kH,kW)
            self.kernel = self.kernel.repeat((self.channels,1,1,1,1))


    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
                                  shape: N,oC,iH,iW
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        #Create gaussian kernel
        # shape: oC,iC/groups,kH,kW
        kernel = self.kernel.to(input.device)

        if self.dim == 2:
            # compute the padding so that the output is always the same size as input
            #stride is 1 so this can be ignored
            pad_height = int(((input.shape[2] - 1) + self.kernel_size[0] - input.shape[2])
                            / 2)
            pad_width  = int(((input.shape[3] - 1) + self.kernel_size[1] - input.shape[3])
                            / 2)

            # add the padding to the input in reflective way
            # (left,right,top,bottom)
            pad_shape = (pad_width, pad_width, pad_height, pad_height)
            input = F.pad(input, pad_shape, mode='reflect')

            # convolve kernel over input
            x_smooth = F.conv2d(input, kernel, groups = self.channels)

        if self.dim == 3:
            # compute the padding so that the output is always the same size as input
            #stride is 1 so this can be ignored
            pad_height = int(((input.shape[2] - 1) + self.kernel_size[0] - input.shape[2])
                            / 2)
            pad_width = int(((input.shape[3] - 1) + self.kernel_size[1] - input.shape[3])
                            / 2)
            pad_depth = int(((input.shape[4] - 1) + self.kernel_size[2] - input.shape[4])
                            / 2)

            # add the padding to the input in reflective way
            # (left, right, top, bottom, front, back)
            pad_shape = (pad_width, pad_width,
                         pad_height, pad_height,
                         pad_depth, pad_depth)
            input = F.pad(input, pad_shape, mode='constant')

            # convolve kernel over input
            x_smooth = F.conv3d(input, kernel, groups = self.channels)
        return x_smooth

def create_identity(cover_area = (-1,+1), grid_size = None, coordinate_type = 'xy', device = 'cpu'):
    """
        Identity x is a transformation that when composed to an input it returns the same input.

        input args:
            cover_area: The boundaries of the input that is to be transformed
                shapes: (x_1,y_1) or (x_0, y_0, x_1, y_1)
            grid_size: The boundaries of the image transformation matrix function
                shapes: (x_1,y_1)
            coordinate_type: xy (width, height) or ij (height,width) coordinates
        output:
            identity transform
            bounds (bounds for each dimension)
    """
    if coordinate_type not in ['xy','ij']:
        raise ValueError("coordinate_type must be xy or ij")

    if len(cover_area) == 2:
        # x_0, y_0, x_1, y_1
        border = 0, 0, cover_area[0], cover_area[1]
    if len(cover_area) == 3:
        # x_0, y_0, z_0, x_1, y_1, z_1
        border = 0, 0, 0, cover_area[0], cover_area[1], cover_area[2]
    else:
        # x_0, y_0, x_1, y_1
        border = cover_area

    dim = (int)(len(border)/2)
    if grid_size is None:
        grid_size = [cover_area[i] for i in range(dim)]

    linear_base = [torch.linspace(border[i], border[dim + i], grid_size[i])
                   for i in range(dim)]

    deformer = torch.stack(
        torch.torch.meshgrid(*linear_base)
    ).unsqueeze(0).float().to(device)

    if coordinate_type == 'xy':
        return deformer.flip(1), border
    else:
        return deformer, cover_area

def standardise(grid, bounds, center = True):
    """
        normalise the grid coordinates based on the input dimensions
        (-1,-1) is the top left pixel, (1,1) is the top right pixel
    """
    grid = grid.clone()
    dims = grid.shape[-1]

    if len(bounds) > dims:
        bounds = [(bounds[i] + bounds[i+dims])/2 for i in range(dims)]
    for i in range(dims):
        # if already standardised
        if bounds[i] == 0:
            continue

        if center:
            grid[...,i] = grid[...,i] - bounds[i]

        grid[...,i] = grid[...,i]/bounds[i]

    return grid

def unstandardise(grid, bounds, center=True):
    """
        normalise the grid coordinates based on the input dimensions
        (-1,-1) is the top left pixel, (1,1) is the top right pixel
    """
    grid = grid.clone()
    dims = grid.shape[-1]

    if len(bounds) > dims:
        bounds = [(bounds[i] + bounds[i+dims])/2 for i in range(dims)]
    for i in range(dims):

        grid[...,i] = grid[...,i]*bounds[i]

        if center:
            grid[...,i] = grid[...,i] + bounds[i]

    return grid


def compose_deformation(phi_1, phi_2, bounds, clamp_sample_grid = True):
    """
        The composition of two deformation fields, it is computed by sampling one deformation field by another.

        input args:
            phi_1: sampled grid
                shapes (N, C, H_in, W_in)
            phi_2: sample locations
                shapes (N, H_out, W_out, 2) or (N, 2, H_out, W_out)
            bounds: bounds of phi_2, this allows you to specify the value range for the sample grid,
                    it is required because the phi_2 has to be constrained to the {-1,+1} domain.
                    Note that this value can not be derived from phi_1 because if it is a function
                    it is not necesarily constrained by the boundaries of the input
                    to which it will be applied e.g. phi_1 might sample 10 to the left, if you constrain
                    phi_2 based on these boundaries your image will be 10 smaller
            output: new transformation phi
                shapes: (N, C, H_out, w_out)

    """
    if (phi_2.shape[1] == 2) or (phi_2.shape[1] == 3):
        # if shape is (N, 2, H_out, W_out) conversion is done automatic
        # to shape (N, H_out, W_out, 2)
        phi_2 = phi_2.transpose(1, -1)

    sampler = standardise(phi_2, bounds)
    return F.grid_sample(phi_1, sampler, padding_mode = 'border', mode ='bilinear', align_corners = True)
