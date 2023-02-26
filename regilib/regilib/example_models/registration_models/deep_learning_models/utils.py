import torch
import pyvista as pv

def cat_tensor_to_batch_tensor(cat_tensor, batches):
    """
    Convert a concatenated tensor into a tensor of shape
    (B*N)xd -> BxNxd

    Note: this function assumes that all batches have same N
    """

    bins = batches.bincount()

    # if only one batch
    if bins.shape[0] == 1:
        return cat_tensor[None]

    # if multiple batches
    list_tensor = torch.split(cat_tensor, bins.tolist())
    return torch.stack(list_tensor)

def batch_tensor_to_cat_tensor(batch_tensor):
    """
    Convert a split tensor into a tensor of shape
    BxNxd -> (B*N)xd

    Note: this function assumes that all batches have same N
    """

    list_tensor = list(torch.unbind(batch_tensor))
    return torch.cat(list_tensor)

def tensor_to_vtk(verts, faces):

    if faces.shape[0] != 3:
        faces = torch.t(faces)

    vertices = verts.cpu().detach().numpy()

    face_size = torch.ones(1, faces.shape[1])*3
    faces = torch.cat((face_size.long(), faces.cpu().detach()))
    faces = faces.permute(1,0).reshape(-1,).numpy()
    return pv.PolyData(vertices, faces)

def compute_jacobian(phi, bounds, grid_size):
    def first_order_derivative(phi):
        # distance between grid points
        delta_i = bounds[0]/(grid_size[0]-1)
        delta_j = bounds[1]/(grid_size[1]-1)
        delta_k = bounds[2]/(grid_size[2]-1)

        # f(x+1) - f(x)
        di = (phi[:,:-1,:,:] - phi[:,1:,:,:])/delta_i
        dj = (phi[:,:,:-1,:] - phi[:,:,1:,:])/delta_j
        dk = (phi[:,:,:,:-1] - phi[:,:,:,1:])/delta_k

        # no information about last entries so zero pad
        di = di[:,:,:-1,:-1]
        dj = dj[:,:-1,:,:-1]
        dk = dk[:,:-1,:-1,:]

        return di, dj, dk

    return first_order_derivative(phi[0])

def compute_3x3_determinant(matrix):
    di, dj, dk = matrix
    di_i, di_j, di_k = di[0], di[1], di[2]
    dj_i, dj_j, dj_k = dj[0], dj[1], dj[2]
    dk_i, dk_j, dk_k = dk[0], dk[1], dk[2]

    return di_i*(dj_j*dk_k-dj_k*dk_j)-di_j*(dj_i*dk_k-dj_k*dk_i)+di_k*(dj_i*dk_j-dj_j*dk_i)
