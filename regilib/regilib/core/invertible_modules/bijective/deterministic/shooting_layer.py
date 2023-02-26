#!/usr/bin/env python3

import torch
from pykeops.torch import LazyTensor
from regilib.core.dynamics.dynamical_state import DynamicalState
from regilib.core.invertible_modules import InvertibleModule
from torchdyn.numerics.odeint import odeint, str_to_solver

class ShootingLayer(torch.nn.Module):
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
        self.solver = solver if type(solver) != str else str_to_solver(solver)

    def gram_matrix(self, x, y):
        """Computes the gram matrix using Gaussian kernels 
        $$
        K(x, y) = \exp \left( \frac{\lVert x - y\rVert_2^2}{\sigma_k^2} \right).
        $$

        :param x: array-like, shape=[., d]
            Row vectors.
        :param y: array-like, shape=[., d]
            Column vectors.

        """
        rows = LazyTensor(x[:, :, None, :])
        cols = LazyTensor(y[:, None, :, :])

        return (-((rows - cols)**2).sum(-1)/(self.sigma**2)).exp()

    def velocity(self, xs, cs, ms):
        """Computes the velocity of a particle given the control points and momenta. 
        $$
        v_t(x) = \sum_{k=1}^{N_{cp}} K(x, c_k(t))\alpha_k(t),
        $$

        :param xs: array-like, shape=[N, d]
            Particle position.
        :param cs: array-like, shape=[C, d]
            Control points position.
        :param ms: array-like, shape=[C, d]
            Control points momentum.
        :returns: array-like, shape=[N, d]
            The velocity for each particle.

        """
        # TODO: add batch support
        xs, cs, ms = xs[None], cs[None], ms[None]

        Vs = []
        for i in range(xs.shape[0]):
            x, c, m = xs[i:i+1], cs[i:i+1], ms[i:i+1]

            K = self.gram_matrix(x, c)
            Km = (K @ m)
            Vs.append(Km)

        return torch.cat(Vs)

    def hamiltonian_eqs(self, x, m):
        """Computes the change in position and change in momentum based
        on hamilton's equations
        $$
        \mathcal{H} = \alpha(0)^TK(c(t),c(t))\alpha(0),
        $$

        :param x: array-like, shape=[N, d]
            Particles.
        :param m: array-like, shape=[N, d]
            Momenta.
        :returns: array-like, shape=[2, N, d]
            $$
            \begin{align*}
            \dot{c}(t) &= \frac{\partial \mathcal{H}}{ \partial \alpha(t)},   &   \dot{\alpha}(t)
            &= -\frac{\partial \mathcal{H}}{ \partial c(t)}
            \end{align*}
            $$
        """
        Km = self.velocity(x, x, m)
        H = 0.5*torch.sum(m * Km, (-2, -1))
        dx_t, dm_t = torch.autograd.grad(H.sum(), (m, x), create_graph=True)
        return torch.stack([dx_t, -dm_t])

    def inner_product_matrix(self, base_point=None):
        """Inner product matrix at the tangent space at a base point.

        :param base_point: Not used (assumed to be identity)
        :returns: array-like, shape=[C, d, d]
            Inner product matrix

        """

        return self.gram_matrix(
            self.control_points[None], self.control_points[None])

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Inner product between two tangent vectors at a base point.

        :param tangent_vec_a: array-like, shape=[C, d]
            Tangent vector at identity base point.
        :param tangent_vec_b: array-like, shape=[C, d]
            Tangent vector at identity base point.
        :param base_point: Not used (assumed to be identity)

        :returns: array-like, shape=[...,]
            Inner product

        """

        inner_prod_mat = self.inner_product_matrix()

        Km = inner_prod_mat @ tangent_vec_b
        H = 0.5*torch.sum(tangent_vec_a * Km, (-2, -1))

        return H.to(tangent_vec_a.device)

    def exp(self, tangent_vec, t0, t1, steps, control_points):
        """Computes the exponential map starting from base point ID and given tantent vector
        $$
        Exp_{ID}: T_{ID}\mathcal{M} \rightarrow \mathcal{M}
        $$

        :param tangent_vec: tensor, shape=[B*N,3]
            Momentum vector that parametrises the deformation
        :param t0: int
            Start time 
        :param t1: int
            End time 
        :param steps: int
            Number of time steps 
        :param control_points: 
            Control points that parametrises the deformation 
        :returns: tuple (array-like, array-like), shape=[steps],[steps, C, d]
            Evaluation time steps and control point integration steps. 
        """

        steps = self.default_n_steps if steps is None else steps
        t_span = torch.linspace(t0, t1, steps)

        # integrate control points
        m0 = tangent_vec.clone()
        c0 = (self.control_points.clone()
              if control_points is None else control_points)
        s0 = torch.stack([c0, m0])
        s_time_steps, st = odeint(
            lambda _, s: self.hamiltonian_eqs(s[0], s[1]),
            s0, t_span, solver=self.solver
        )

        return s_time_steps, st

    def forward(self, momentum, t0=None, t1=None, steps=None,
                include_trajectory=False, control_points=None, **kwargs):
        """Computes the exponential map starting from base point ID and given tantent vector
        $$
        Exp_{ID}: T_{ID}\mathcal{M} \rightarrow \mathcal{M}
        $$

        :param momentum: tensor, shape=[B*N,3]
            Momentum vector that parametrises the deformation
        :param t0: int
            Start time (default: 0)
        :param t1: int
            End time (default: 1)
        :param steps: int
            Number of time steps (default: self.default_n_steps)
        :param include_trajectory: bool
            Whether to include the control point trajectory in the dynamic state.
        :param control_points: 
            Control points that parametrises the deformation (default: self.control_points)
        :returns: tuple (array-like, array-like), shape=[steps],[steps, C, d]
            Evaluation time steps and control point integration steps. 
        """

        t0 = ShootingLayer.DEFAULT_T0 if t0 is None else t0
        t1 = ShootingLayer.DEFAULT_T1 if t1 is None else t1
        steps = self.default_n_steps if steps is None else steps

        control_points = (self.control_points if control_points is None
                          else control_points)

        # t: 0 -> 1
        return self.exp(
            momentum, t0, t1, steps, control_points)

