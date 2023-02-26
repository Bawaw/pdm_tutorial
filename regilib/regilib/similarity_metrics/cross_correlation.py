import torch
import torch.nn.functional as func

class NCC:
    def __call__(self, F, M):
        n_batches = F.shape[0]
        f, m = F.view(n_batches, -1), M.view(n_batches, -1)

        return ((f-f.mean())*(m-m.mean())/(f.std()*m.std())).mean(1)

class LNCC:
    def __init__(self, kernel_size = (25,25,25), eps = 1.e-15, sigma=0.1):
        self.kernel_size = torch.tensor(kernel_size)
        self.eps, self.sigma = eps, sigma

    def __call__(self, F, M):
        # implementation based on VoxelMorph

        neigh_size = self.kernel_size.prod()
        kernel = torch.ones([1,1,*self.kernel_size]).to(M.device)

        # compute local means and sums
        f_local_sum = func.conv3d(F, kernel)
        m_local_sum = func.conv3d(M, kernel)
        fm_local_sum = func.conv3d(F*M, kernel)
        f2_local_sum = func.conv3d(F**2, kernel)
        m2_local_sum = func.conv3d(M**2, kernel)

        f_local_mean = f_local_sum/neigh_size
        m_local_mean = m_local_sum/neigh_size

        #((a-a_bar)(b-b_bar))^2 = (a*b - a*b_bar - a_bar*b + a_bar*b_bar)^2
        cor = (fm_local_sum - f_local_mean*m_local_sum
               - m_local_mean*f_local_sum
               + f_local_mean*m_local_mean*neigh_size)

        # (a-a_bar)^2 = a^2 - 2a*a_bar + a_bar^2
        std_m = (m2_local_sum
                 - 2*m_local_mean*m_local_sum
                 + m_local_mean**2*neigh_size)
        std_f = (f2_local_sum
                 - 2*f_local_mean*f_local_sum
                 + f_local_mean**2*neigh_size)

        # result
        lncc = cor * cor / (std_m * std_f + self.eps)

        # reduce the dimensions
        reduce_dims = list(range(1, M.dim()))
        lncc = 1 - lncc.mean(reduce_dims)

        # divide by balancing constant (as seen in mermaid)
        return lncc / (self.sigma**2)
