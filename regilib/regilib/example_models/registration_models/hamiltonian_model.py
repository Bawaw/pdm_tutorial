import torch
import random
import torch.optim as optim
from torch import Tensor
import geomstats.backend as gs
import torch.nn.functional as F
from pykeops.torch import LazyTensor
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.riemannian_metric import RiemannianMetric
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from regilib.similarity_measures.cross_correlation import LNCC
from regilib.registration_models.registration_model import RegistrationModel
from regilib.numerics.integrator import EulerIntegrator

class HamiltonianShapeManifold(RiemannianMetric, RegistrationModel):
    def __init__(self, control_points, sigma,
                 device='cpu',
                 n_time_steps=11,
                 integrator=EulerIntegrator(),
                 log_optimiser=optim.Adam,
                 log_iters=5000,
                 log_step_size=0.01,
                 log_step_size_patience=5,
                 log_patience=20,
                 log_epsilon=1.e-3,
                 **kwargs):

        self.device = device
        self.n_time_steps = n_time_steps
        self.integrator = integrator
        self.log_optimiser = log_optimiser
        self.log_iters = log_iters
        self.log_patience = log_patience
        self.log_epsilon = log_epsilon
        self.log_step_size = log_step_size
        self.log_step_size_patience = log_step_size_patience

        self.sigma = sigma.clone().to(device)
        self.control_points = self.dim_match(
            control_points.clone(), 3).contiguous()

        manifold_dim = (self.control_points.shape[1] *
                        self.control_points.shape[2])
        super().__init__(manifold_dim, **kwargs)

    def gramm_matrix(self, x, y):
        """Gaussian kernel matrix """

        rows = LazyTensor(x[:, :, None, :])
        cols = LazyTensor(y[:, None, :, :])

        return (-((rows - cols)**2).sum(-1)/(self.sigma**2)).exp()

    def velocity(self, xs, ys, ms):
        """Kernel product Km"""
        xs = xs.contiguous()

        Vs = []
        for i in range(xs.shape[0]):
            x, m, y = xs[i:i+1], ms[i:i+1], ys[i:i+1]

            K = self.gramm_matrix(x,y)
            Km = (K @ m)
            Vs.append(Km)

        return torch.cat(Vs)

    def hamiltonian_eqs(self, x, m):
        Km = self.velocity(x, x, m)
        H = 0.5*torch.sum(m * Km, (-2, -1))
        dx_t, dm_t = torch.autograd.grad(H.sum(), (m, x), create_graph=True)
        return torch.stack([dx_t, -dm_t])

    def inner_product_matrix(self, base_point=None):
        """Inner product matrix at the tangent space at a base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        return self.gramm_matrix(self.control_points, self.control_points)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[..., dim]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point: array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        ta =tangent_vec_a.clone().to(self.device)
        ta = self.unvectorise(ta, (-1, 3))

        tb = tangent_vec_b.clone().to(self.device)
        tb = self.unvectorise(tb, (-1, 3))

        if base_point is not None:
            bp = base_point.clone().to(self.device)
            bp = self.unvectorise(bp, self.input_shape)
        else:
            bp = None

        inner_prod_mat = self.inner_product_matrix(bp)

        Km = inner_prod_mat @ tb
        H = 0.5*torch.sum(ta * Km, (-2, -1))

        return H.to(tangent_vec_a.device)

    def squared_dist_by_deformation(self, deformation_params):
        return self.squared_norm(vector=deformation_params, base_point=None)

class HamiltonianShapeManifoldMeshMetric(HamiltonianShapeManifold):
    """
    Imposes a metric on the mesh shape manifold
    """

    def __init__(self, loss=None, **kwargs):
        if loss is None:
            self.loss = lambda fixed, f_star: torch.abs(fixed-f_star).sum(-1).mean(1)
        else:
            self.loss = loss

        self.input_shape = (-1, 3)
        super().__init__(**kwargs)

    def exp(self, tangent_vec, base_point, point_type=None):
        """
        Exponential map

        ..math::
            Exp_x: T_x\mathcal{M} \rightarrow \mathcal{M}

        Parameters
        ----------
        tangent_vec : tensor, shape=[B,C,3] or [C,3]
            momentum vector that parametrises the deformation
        base_point : tensor, shape = [B,C,3] or [C,3]
            shape x, to be deformed over time

        Returns
        -------
        target_point : tensor, shape=[,]
            integrated base point
        """

        _tangent_vec, _base_point = tangent_vec, base_point

        # TODO this is a temp fix, to discriminate between batched vector and point-cloud
        if tangent_vec.shape[-1] != 3:
            # vector format to tangent_vec format
            tangent_vec = self.unvectorise(tangent_vec, self.input_shape)
        if base_point.shape[-1] != 3:
            # vector format to volume format
            base_point = self.unvectorise(base_point, self.input_shape)

        tangent_vec = self.dim_match(tangent_vec.clone(), 3)
        base_point = self.batch_like(base_point.clone(), tangent_vec)

        # integrate control points
        c0 = self.batch_like(self.control_points.clone(), tangent_vec)
        m0 = tangent_vec.clone()
        s0 = torch.stack([c0, m0])

        ds_dt = lambda _, s: self.hamiltonian_eqs(s[0], s[1])
        t = torch.linspace(0, 1, self.n_time_steps+1, device=self.device)
        states = self.integrator(ds_dt, s0, t)

        # warp input shape
        x0 = base_point.clone()
        dx_dt = lambda i, x: self.velocity(x, states[i,0], states[i,1])
        Xs = self.integrator(dx_dt, x0, t)
        warped = Xs[-1]

        if _base_point.shape[-1] != 3:
            warped = self.vectorise(warped, batched=True)

        return self.batch_like(warped, _base_point).to(_base_point.device)

    def log(self, point, base_point, point_type=None):
        """
        Logarithm map

        ..math::
            Log : \mathcal{M} \times \mathcal{M} \rightarrow T_x\mathcal{M}

        Parameters
        ----------
        point : tensor, shape=[B,C*3] or [C*3]
            target shape
        base_point : tensor, shape = [B,C*3] or [1,C*3] or [C*3]
            shape to be deformed

        Returns
        -------
        tangent_vec : tensor, shape=[,]
            tangent_vec parametrising the deformation between base_point and point
        """

        # TODO this is a temp fix, to discriminate between batched vector and point-cloud
        if point.shape[-1] != 3:
            # convert vector format to point cloud format
            fixed = self.unvectorise(point.clone(), self.input_shape)
        else:
            fixed = point.clone()

        if base_point.shape[-1] != 3:
            moving = self.unvectorise(base_point.clone(), self.input_shape)
        else:
            moving = base_point.clone()


        # shapes to be deformed
        fixed, moving = self.dim_match(fixed, 3), self.dim_match(moving, 3)

        # BATCH | CP_SHAPE
        momenta_shape = fixed.shape[0:1] + self.control_points.shape[1:]
        momenta = torch.zeros(
            momenta_shape, device=self.device, requires_grad=True, dtype=point.dtype)

        # log optimiser
        optimiser = self.log_optimiser([momenta], lr=self.log_step_size)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, 'min', patience=self.log_step_size_patience, factor=0.5)

        steps_since_plat, last_plat = 0, 0
        for k in range(self.log_iters):
            optimiser.zero_grad()

            f_star = self.exp(momenta, moving)
            dist = self.loss(fixed, f_star)
            print(dist.mean())
            reg = self.inner_product(momenta, momenta, self.control_points)
            step_loss = dist + reg
            step_loss.sum().backward()
            optimiser.step()
            scheduler.step(step_loss.mean())

            # if stuck on minimum, stop
            delta_loss = abs(last_plat - step_loss.mean().data)
            if ((steps_since_plat >= self.log_patience) and
                (delta_loss <= self.log_epsilon)):
                break
            elif abs(last_plat - step_loss.mean().data) > self.log_epsilon:
                last_plat, steps_since_plat = step_loss.mean(), 0
            steps_since_plat += 1

        return momenta.view(momenta.shape[0], -1).detach().to(point.device)

    def __call__(self, fixed, moving, return_transform=False):
        momentum = self.log(fixed.pos, moving.pos)
        f_star = self.exp(momentum, moving.pos)

        if return_transform:
            return Data(pos=f_star, face=moving.face.clone()), momentum

        return Data(pos=f_star, face=moving.face.clone())

class HamiltonianShapeManifoldVolumeMetric(HamiltonianShapeManifold):
    """
    Imposes a metric on the volumetric shape manifold
    """

    def __init__(self, input_shape, loss = None, **kwargs):
        if loss is None:
            from regilib.similarity_measures.cross_correlation import LNCC
            self.loss = LNCC()
        else:
            self.loss = loss

        self.input_shape = input_shape
        super().__init__(**kwargs)

    def create_identity(dims, device='cpu'):
        linear_base = [torch.linspace(-1, 1, d) for d in dims]

        identity_grid = torch.stack(
            torch.torch.meshgrid(*linear_base)
        )[None].to(device)

        axes = torch.arange(0,identity_grid.dim()).tolist()

        # move coordinates to the end
        axes.append(axes.pop(1))

        return identity_grid.permute(axes).flip(-1)

    def exp(self, tangent_vec, base_point, point_type=None, time_steps = False):
        """
        Exponential map

        ..math::
            Exp_x: T_x\mathcal{M} \rightarrow \mathcal{M}

        Parameters
        ----------
        tangent_vec : tensor, shape=[B,C,3] or [C,3]
            momentum vector that parametrises the deformation
        base_point : tensor, shape = [B, C, D, W, H] or [C,3]
            shape x, to be deformed over time

        Returns
        -------
        target_point : tensor, shape=[,]
            integrated base point
        """

        _tangent_vec, _base_point = tangent_vec.clone(), base_point.clone()

        # vector format to tangent_vec format
        tangent_vec = self.unvectorise(tangent_vec, (-1, 3))
        # vector format to volume format
        base_point = self.unvectorise(base_point, self.input_shape)

        tangent_vec = self.dim_match(tangent_vec, 3)
        base_point = self.match_quantity(
            self.dim_match(base_point.clone(), 5), tangent_vec)
        n_batches = tangent_vec.shape[0]

        # integrate control points, backwards in time
        c0 = self.batch_like(self.control_points.clone(), tangent_vec)
        m0 = -tangent_vec.clone()
        s0 = torch.stack([c0, m0])

        ds_dt = lambda _, s: self.hamiltonian_eqs(s[0], s[1])
        t = torch.linspace(0, 1, self.n_time_steps+1, device=self.device)
        states = self.integrator(ds_dt, s0, t)

        if time_steps:
            return states

        # integrate identity grid
        identity_grid = HamiltonianShapeManifoldVolumeMetric.create_identity(
            base_point.shape[-3:], self.device)
        identity_grid = self.batch_like(identity_grid, base_point)

        deformation_field = identity_grid.view(n_batches, -1, 3)
        dx_dt = lambda i, x: self.velocity(x, states[i,0], states[i,1])
        phi = self.integrator(dx_dt, deformation_field, t)

        # warp image
        sample_locations = phi[-1].view(n_batches, *base_point.shape[2:], 3)
        base_point = F.grid_sample(base_point, sample_locations, padding_mode='border',
                                   mode='bilinear', align_corners=True)

        # if base_point was in vector format
        if _base_point.dim() <= 2:
            base_point = self.vectorise(base_point, batched=True)
        return self.batch_like(base_point, _base_point).to(_base_point.device)

    def log(self, point, base_point, point_type=None):
        """
        Logarithm map

        ..math::
            Log : \mathcal{M} \times \mathcal{M} \rightarrow T_x\mathcal{M}

        Parameters
        ----------
        point : tensor, shape=[B,3,d,h,w] or [3,d,h,w]
            target shape
        base_point : tensor, shape = [B,3,d,h,w] or [1,3,d,h,w] or [3,d,h,w]
            shape to be deformed

        Returns
        -------
        tangent_vec : tensor, shape=[,]
            tangent_vec parametrising the deformation between base_point and point
        """

        # convert vector format to volume format
        fixed = self.unvectorise(point.clone(), self.input_shape)
        moving = self.unvectorise(base_point.clone(), self.input_shape)

        # shapes to be deformed
        fixed, moving = self.dim_match(fixed, 5), self.dim_match(moving, 5)

        # BATCH | CP_SHAPE
        momenta_shape = fixed.shape[0:1] + self.control_points.shape[1:]
        momenta = torch.zeros(momenta_shape, device=self.device, requires_grad=True)

        # log optimiser
        optimiser = self.log_optimiser([momenta], lr=self.log_step_size)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, 'min', patience=self.log_step_size_patience, factor=0.5)

        steps_since_plat, last_plat = 0, 0
        for k in range(self.log_iters):
            optimiser.zero_grad()

            f_star = self.exp(momenta, moving)
            dist = self.loss(fixed, f_star)
            reg = self.inner_product(momenta, momenta, self.control_points)
            print(dist.mean().item())
            step_loss = dist + reg
            step_loss.sum().backward()
            optimiser.step()
            scheduler.step(step_loss.mean())

            # if stuck on minimum, stop
            delta_loss = abs(last_plat - step_loss.mean().data)
            if ((steps_since_plat >= self.log_patience) and
                (delta_loss <= self.log_epsilon)):
                break
            elif abs(last_plat - step_loss.mean().data) > self.log_epsilon:
                last_plat, steps_since_plat = step_loss.mean(), 0
            steps_since_plat += 1

        return momenta.detach().view(momenta.shape[0], -1).to(point.device)

    def __call__(self, fixed, moving, return_transform=False):
        momentum = self.log(fixed.vol, moving.vol)
        f_star = self.exp(momentum, moving.vol)

        if return_transform:
            return Data(vol=f_star), momentum

        return Data(vol=f_star)
