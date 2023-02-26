#!/usr/bin/env python3

import torch

from pykeops.torch import LazyTensor
from regilib.core.invertible_modules import InvertibleModule
from regilib.core.dynamics.dynamical_state import DynamicalState
from torchdyn.numerics.odeint import odeint, odeint_mshooting, str_to_solver

class ShootingLayer(InvertibleModule):
    DEFAULT_T0 = 0
    DEFAULT_T1 = 1

    def __init__(
            self, control_points, sigma, default_n_steps=5,
            solver='rk4', **kwargs):

        super().__init__()
        self.sigma = sigma
        self.default_n_steps = default_n_steps
        self.control_points = torch.nn.parameter.Parameter(
            control_points, requires_grad=True)
        # self.control_points = torch.nn.parameter.Parameter(
        #     control_points, requires_grad=False)
        self.solver = solver if type(solver) != str else str_to_solver(solver)

    def gramm_matrix(self, x, y):
        """Gaussian kernel matrix """

        rows = LazyTensor(x[:, :, None, :])
        cols = LazyTensor(y[:, None, :, :])

        return (-((rows - cols)**2).sum(-1)/(self.sigma**2)).exp()

    def velocity(self, xs, cs, ms):
        """Kernel product Km"""
        #xs = xs.contiguous()
        # TODO: add batch support
        xs, cs, ms = xs[None], cs[None], ms[None]

        Vs = []
        for i in range(xs.shape[0]):
            x, c, m = xs[i:i+1], cs[i:i+1], ms[i:i+1]

            K = self.gramm_matrix(x, c)
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
        return self.gramm_matrix(self.control_points[None], self.control_points[None])

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
        inner_prod_mat = self.inner_product_matrix()

        Km = inner_prod_mat @ tangent_vec_b
        H = 0.5*torch.sum(tangent_vec_a * Km, (-2, -1))

        return H.to(tangent_vec_a.device)

    def squared_dist_by_deformation(self, deformation_params):
        return self.squared_norm(vector=deformation_params, base_point=None)

    def exp(self, tangent_vec, t0=0, t1=1,
            steps=None, control_points=None):
        """
        Exponential map

        ..math::
            Exp_{ID}: T_{ID}\mathcal{M} \rightarrow \mathcal{M}

        Parameters
        ----------
        tangent_vec (m_0): tensor, shape=[B*N,3]
            momentum vector that parametrises the deformation
        control_points (c_0): tensor, shape=[B*N,3]
            control points that parametrises the deformation

        Returns
        -------
        control_point_state (s_t): tensor, shape=[T, 2, B*N, 3]
            state of the system of control points at each time step t
        """

        steps = self.default_n_steps if steps is None else steps
        t_span = torch.linspace(t0, t1, steps)

        # integrate control points
        m0 = tangent_vec.clone()
        c0 = (self.control_points.clone()
              if control_points is None else control_points)
        s0 = torch.stack([c0, m0])
        _, st = odeint(
            lambda _, s: self.hamiltonian_eqs(s[0], s[1]),
            s0, t_span, solver=self.solver
        )

        return st

        # # warp input shape
        # # TODO: make this work with adaptive solver
        # x0 = base_point.to(m0.device)

        # t_step = abs(t1-t0)/(steps-1)

        # Xs_t, Xs = odeint(
        #     lambda t, x: self.velocity(x, states[
        #         int(t/t_step), 0], states[int(t/t_step), 1])[0],
        #     x0, t_span, solver=self.solver
        # )

        # if cp_states: return Xs, states
        # return Xs

    def warp_pc(self, x0, states, t0, t1, steps):
        t_span = torch.linspace(t0, t1, steps)

        # warp input shape
        t_step = abs(t1-t0)/(steps-1)

        Xs_t, Xs = odeint(
            lambda t, x: self.velocity(x, states[
                int(t/t_step), 0], states[int(t/t_step), 1])[0],
            x0, t_span, solver=self.solver
        )

        return Xs


    def forward(self, ds):
        z_dot = None

        return z_dot

    # def forward(self, ds_z, momentum, t0=None, t1=None, steps=None,
    #             include_trajectory=False, **kwargs):
    #     ds_x = ds_z.clone() if isinstance(
    #         ds_z, DynamicalState) else DynamicalState(state=ds_z)
    #     t0 = ShootingLayer.DEFAULT_T0 if t0 is None else t0
    #     t1 = ShootingLayer.DEFAULT_T1 if t1 is None else t1
    #     steps = self.default_n_steps if steps is None else steps

    #     # t: 0 -> 1
    #     st = self.exp(momentum, t0, t1, steps)
    #     xt = self.warp_pc(ds_x.state, st, t0, t1, steps)
    #     ds_x.state = xt[-1]
    #     if include_trajectory: ds_x.append_or_create('trajectory', xt)

    #     return ds_x

    # def inverse(self, ds_x, momentum, t0=None, t1=None, steps=None,
    #             include_trajectory=False, control_points=None, **kwargs):
    #     ds_z = ds_x.clone() if isinstance(
    #         ds_x, DynamicalState) else DynamicalState(state=ds_x)
    #     t0 = ShootingLayer.DEFAULT_T1 if t0 is None else t0
    #     t1 = ShootingLayer.DEFAULT_T0 if t1 is None else t1
    #     steps = self.default_n_steps if steps is None else steps

    #     # first integrate control points in forward direction
    #     cps = (self.exp(self.control_points, momentum, None, t1, t0, steps)
    #            if control_points is None else control_points)

    #     # t: 1 -> 0
    #     Zs = self.exp(ds_z.state, momentum, cps[-1], t0, t1, steps)
    #     ds_z.state = Zs[-1]
    #     if include_trajectory: ds_z.append_or_create('trajectory', Zs)

    #     return ds_z

    # def forward(self, ds_z, momentum, t0=None, t1=None, steps=None,
    #             include_trajectory=False, control_points=None, **kwargs):
    #     ds_x = ds_z.clone() if isinstance(
    #         ds_z, DynamicalState) else DynamicalState(state=ds_z)
    #     t0 = ShootingLayer.DEFAULT_T0 if t0 is None else t0
    #     t1 = ShootingLayer.DEFAULT_T1 if t1 is None else t1
    #     steps = self.default_n_steps if steps is None else steps

    #     # t: 0 -> 1
    #     cps = (self.control_points
    #            if control_points is None else control_points)
    #     Xs = self.exp(ds_x.state, momentum, cps,
    #                   t0, t1, steps)
    #     ds_x.state = Xs[-1]
    #     if include_trajectory: ds_x.append_or_create('trajectory', Xs)

    #     return ds_x

    # def inverse(self, ds_x, momentum, t0=None, t1=None, steps=None,
    #             include_trajectory=False, control_points=None, **kwargs):
    #     ds_z = ds_x.clone() if isinstance(
    #         ds_x, DynamicalState) else DynamicalState(state=ds_x)
    #     t0 = ShootingLayer.DEFAULT_T1 if t0 is None else t0
    #     t1 = ShootingLayer.DEFAULT_T0 if t1 is None else t1
    #     steps = self.default_n_steps if steps is None else steps

    #     # first integrate control points in forward direction
    #     cps = (self.exp(self.control_points, momentum, None, t1, t0, steps)
    #            if control_points is None else control_points)

    #     # t: 1 -> 0
    #     Zs = self.exp(ds_z.state, momentum, cps[-1], t0, t1, steps)
    #     ds_z.state = Zs[-1]
    #     if include_trajectory: ds_z.append_or_create('trajectory', Zs)

    #     return ds_z
