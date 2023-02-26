import torch
import torch.optim as optim

class RegistrationModel:
    def dim_match(self, input, dim, requires_grad=True):
        dim_diff = input.dim() - dim
        if dim_diff < 0:
            new_shape = torch.Size([abs(dim_diff)]) + input.shape
            input = input.reshape(new_shape)

        # only downscale if the batch dimension is 1
        elif dim_diff > 0 and (torch.tensor(input.shape[:dim_diff]) == 1).sum():
            new_shape = input.shape[dim_diff:]
            input = input.reshape(new_shape)
        if input.requires_grad and not requires_grad:
            input = input.detach()
        else:
            input.requires_grad_(requires_grad)
        return input.to(self.device)

    def match_quantity(self, input, other):
        # only add empty dimensions if the first dimension is empty
        if (other.shape[0] > input.shape[0]) and input.shape[0] == 1:
            new_shape = list(other.shape[0:1]) + [1]*(input.dim()-1)
            input = input.repeat(new_shape)
        return input

    def batch_like(self, input, other, requires_grad=None):
        requires_grad = other.requires_grad if requires_grad is None else requires_grad
        input = self.dim_match(input, other.dim(), requires_grad=requires_grad)
        input = self.match_quantity(input, other)
        return input

    def unvectorise(self, input, shape):
        # if batched_vector
        if input.dim() == 2:
            input = input.view(input.shape[0], *shape)

        # if not batched
        elif input.dim() == 1:
            input = input.view(*shape)
        return input

    def vectorise(self, input, batched = True):
        # if batched_vector
        if batched:
            input = input.view(input.shape[0], -1)
        # if not batched
        else:
            input = input.view(-1)
        return input

    def _log_optimiser(self,fixed, moving):
        # BATCH | CP_SHAPE
        momenta_shape = fixed.shape[0:1] + self.control_points.shape[1:]
        momenta = torch.zeros(momenta_shape, device=self.device, requires_grad=True)


        # log optimiser
        optimiser = self.log_optimiser([momenta], lr=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, 'min', patience=5, factor=0.5)

        steps_since_plat, last_plat = 0, 0
        for k in range(self.log_iters):
            optimiser.zero_grad()

            f_star = self.exp(momenta, moving)
            dist = torch.abs(fixed-f_star).sum(-1).mean(1)
            step_loss = dist.sum()
            print(step_loss)
            step_loss.backward()
            optimiser.step()
            scheduler.step(step_loss)

            # if stuck on minimum, stop
            delta_loss = abs(last_plat - step_loss.data)
            if ((steps_since_plat >= self.log_patience) and
                (delta_loss <= self.log_epsilon)):
                break
            elif abs(last_plat - step_loss.data) > self.log_epsilon:
                last_plat, steps_since_plat = step_loss, 0
            steps_since_plat += 1

        return momenta.detach()

    def __call__(self, fixed, moving, return_transform = False):
        raise NotImplementedError

    def squared_dist_by_deformation(self, deformation_params):
        raise NotImplementedError("""No implementation for {}.dist_by_deformation,
        can this registration model be used as a metric?""".format(self))
