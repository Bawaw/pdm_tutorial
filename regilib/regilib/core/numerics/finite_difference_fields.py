import torch


def second_order_central_difference(field, delta, dim, device):
    # see https://mermaid.readthedocs.io/en/latest/_modules/mermaid/finite_differences.html#FD.ddXc
    end = field.shape[dim]
    x_min_h = torch.zeros(field.shape, device=device)
    x_plus_h = torch.zeros(field.shape, device=device)

    # # f(x+h)
    x_plus_h_view = x_plus_h.narrow(dim, 0, end-1)
    x_plus_h_view[:] = field.narrow(dim, 1, end-1)

    # interpolate the missing information
    x_plus_h_view = x_plus_h.narrow(dim, end-1, 1)
    x_plus_h_view[:] = 2*field.narrow(dim, end-1, 1)-field.narrow(dim, end-2, 1)

    # f(x-h)
    x_min_h_view = x_min_h.narrow(dim, 1, end-1)
    x_min_h_view[:] = field.narrow(dim, 0, end-1)

    # interpolate the missing information
    x_min_h_view = x_min_h.narrow(dim, 0, 1)
    x_min_h_view[:] = 2*field.narrow(dim, 0, 1) - field.narrow(dim, 1, 1)

    # (f(x+h) - 2*f(x) + f(x-h))/h**2
    return (x_plus_h - field - field + x_min_h) / (delta**2)

def laplacian(field, delta, device):
    return torch.stack([second_order_central_difference(field, delta[d], 2+d, device)
                        for d in range(field.dim() - 2)])
