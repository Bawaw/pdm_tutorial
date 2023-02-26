import torch
class EulerIntegrator:
    def __call__(self, f, y0, t):
        time_steps = [y0]
        for i in range(t.shape[0]-1):
            dt = t[i+1] - t[i]
            y0 = y0+dt*f(i, y0)
            time_steps.append(y0)

        return torch.stack(time_steps)
