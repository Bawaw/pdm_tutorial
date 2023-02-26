import pyvista as pv
from regilib.utils import adapter
from torch_geometric.data import Data

class PVproDecimate:
    def __init__(self, decimation_percentage=0.7):
        self.decimation_percentage = decimation_percentage
    def __call__(self, mesh):
        assert 'batch' not in mesh.keys or mesh.batch.max == 0, """
            plotter does not accept batched tensors
        """

        assert 'cpu' in mesh.pos.device.type and 'cpu' in mesh.face.device.type, """
            plotter does not accept meshes on GPU
        """

        isTGMesh = isinstance(mesh, Data)
        if isTGMesh:
            mesh = adapter.torch_geomtric_data_to_vtk(mesh.pos, mesh.face)

        mesh = mesh.decimate_pro(self.decimation_percentage)

        if isTGMesh:
            mesh = adapter.vtk_to_torch_geometric_data(mesh)

        return mesh

# class IterativeEdgeCollapse(object):
#     def __init__(self, percentage=0.2):
#         from torch_geometric.transforms import FaceToEdge, Distance
#         self.f2e = FaceToEdge(remove_faces=False)
#         self.compute_edge_dist = Distance(cat=False)
#         self.percentage = percentage
#         self.edge_indices = None

#     def __call__(self, data):
#         data = self.f2e(data)
#         data = self.compute_edge_dist(data)
#         edges = data.edge_index[:, data.edge_attr.flatten() < self.percentage]

#         # face contains e1 -> e2 and e2 -> e1
#         edges = torch.cat([edges, edges.flip(0)], -1)

#         edge_face_mask = (
#             (data.face[:2].T[:,None] == edges.T).all(-1).any(1)
#             + (data.face[1:].T[:,None] == edges.T).all(-1).any(1)
#             + (data.face[::2].T[:,None] == edges.T).all(-1).any(1))

#         i_index, j_index = edges
#         data.face = data.face[:, ~edge_face_mask]
#         breakpoint()
#         data.face[data.face == j_index] = i_index

#         data.edge_index, data.edge_attr = None, None

#         return data

# class IterativeEdgeCollapse(object):
#     def __init__(self, percentage=0.2):
#         from torch_geometric.transforms import FaceToEdge, Distance
#         self.f2e = FaceToEdge(remove_faces=False)
#         self.compute_edge_dist = Distance(cat=False)
#         self.percentage = percentage
#         self.edge_indices = None

#     def __call__(self, data):
#         breakpoint()
#         for i in range(10000):
#             data = self.f2e(data)
#             data = self.compute_edge_dist(data)
#             edge = data.edge_index[:, data.edge_attr.flatten().argmin()]
#             i_index, j_index = edge.min(), edge.max()
#             vertex_mask = torch.ones(data.pos.shape[0])
#             vertex_mask[j_index] = False
#             data.pos = data.pos[vertex_mask.bool()]

#             edge_face_mask = (
#                 # face contains e1 -> e2
#                 (data.face[:2].T == edge).all(-1)
#                 + (data.face[1:].T == edge).all(-1)
#                 + (data.face[::2].T == edge).all(-1)
#                 # face contains e2 -> e1
#                 + (data.face[:2].T == edge.flip(0)).all(-1)
#                 + (data.face[1:].T == edge.flip(0)).all(-1)
#                 + (data.face[::2].T == edge.flip(0)).all(-1))
#             data.face = data.face[:, ~edge_face_mask]

#             data.face[data.face == j_index] = i_index
#             data.face[data.face > j_index] -= 1
#             data.edge_index, data.edge_attr = None, None

#         return data

# class RemoveRandomPoints(object):
#     def __init__(self, num, seed=0):
#         self.num = num
#         self.seed = seed

#     def __call__(self, data):
#         from regilib.utils.adapter import torch_geomtric_data_to_vtk, vtk_to_torch_geometric_data

#         if self.seed is not None: torch.manual_seed(self.seed)
#         indices = torch.randperm(data.pos.shape[0])[:self.num]

#         pv_data = torch_geomtric_data_to_vtk(data.pos, data.face)
#         pv_data, _ = pv_data.remove_points(indices)
#         return vtk_to_torch_geometric_data(pv_data)
