import torch
import numpy as np

from torch_geometric.nn import knn
from torch_geometric.transforms.face_to_edge import FaceToEdge

import regilib.registration_models.deep_learning_models.layers as layers


class FaceOutliner:
    def __init__(self):
        self.face_to_edge = FaceToEdge(remove_faces = False)

    def get_border(self, obj):
        e1 = obj.face.T[:,[0,1]]
        e2 = obj.face.T[:,[1,2]]
        e3 = obj.face.T[:,[0,2]]
        edges = torch.cat([e1,e2,e3])
        edges_ext = torch.cat([edges,edges.roll(1,1)])
        (_, ind, counts) = edges_ext.unique(dim = 0, sorted = False, return_counts = True,
                                            return_inverse = True)
        counts = counts[ind][:edges.shape[0]]

        return edges[counts < 2].flatten()

    def __call__(self, fixed, moving):
        #find the border of the moving shape
        moving_border_verts_ind = self.get_border(moving)

        # find closetst neighbours
        fixed_nearest_in_moving = knn(moving.pos, fixed.pos, 1)[1]

        # if nearest neighbour is border then this is out of bound segment = 0
        # else it is within the segmentation segment = 1
        fixed.segment = torch.tensor(1 - np.isin(fixed_nearest_in_moving.numpy(), moving_border_verts_ind))

        # check if the segment is connected
        #fixed = self.face_to_edge(fixed)
        #(sub_index,_) = subgraph(fixed.segment.bool(), fixed.edge_index)
        #assert not contains_isolated_nodes(sub_index)

        return fixed

class UniformSampler:
    def __init__(self, n = 5000, original_vert_key = None, use_batch_seed = False):
        self.n, self.original_vert_key, self.use_batch_seed = n, original_vert_key, use_batch_seed

    def __call__(self, pc):
        if self.original_vert_key is not None:
            pc[self.original_vert_key] = pc.pos.clone()

        if self.use_batch_seed:
            shuffle = np.random.RandomState(seed=pc.sample_idx).permutation(pc.pos.shape[0])[0:self.n]
        else:
            shuffle = torch.randperm(pc.pos.shape[0])[0:self.n]
        pc.pos = pc.pos[shuffle]
        return pc

class PointSampler(object):
    """
    This code is slightly adapted from
    https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.SamplePoints
    """
    def __init__(self, n_samples, segment_only = False, original_vert_key = None):
        self.n_samples = n_samples
        self.original_vert_key = original_vert_key
        self.segment_only = segment_only

    def __call__(self, data):
        pos, face = data.pos.clone(), data.face.clone()
        assert pos.size(1) == 3 and face.size(0) == 3

        if self.original_vert_key is not None:
            data[self.original_vert_key] = pos

        if face.shape[0] != 3:
            face = torch.t(face)

        if self.segment_only and 'segment' in data.keys:
            segment_ind = data.segment.nonzero().flatten()

            # create map from ind -> new ind
            ind_map = torch.zeros(data.segment.shape[0], dtype = torch.long)
            ind_map[segment_ind] = torch.arange(0, segment_ind.shape[0])

            # vertices that are part of segmented region
            pos = pos[segment_ind]

            # select faces that connect the segmented vertices
            face = face[:, np.isin(face.numpy(), segment_ind).sum(0) == 3]
            face = ind_map[face.view(-1)].view(3, -1)


        pos_max = pos.max()
        pos = pos / pos_max

        area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
        area = area.norm(p=2, dim=1).abs() / 2

        prob = area / area.sum()
        sample = torch.multinomial(prob, self.n_samples, replacement=True)
        face = face[:, sample]

        frac = torch.rand(self.n_samples, 2, device=pos.device)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]

        pos_sampled = pos[face[0]]
        pos_sampled += frac[:, :1] * vec1
        pos_sampled += frac[:, 1:] * vec2

        pos_sampled = pos_sampled * pos_max

        data.pos = pos_sampled

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num)

class Standardise:
    def __init__(self, bounds, key = 'pos', center = True):
        self.bounds, self.center, self.key = bounds, center, key
        print('TODO: standardise has to be cleaned')


    def __call__(self, pc):
        pc[self.key]= layers.standardise(pc[self.key], self.bounds, self.center)
        return pc

class UnStandardise:
    def __init__(self, bounds, key = 'pos', center = True):
        self.bounds, self.center, self.key = bounds, center, key


    def __call__(self, pc):
        pc[self.key]= layers.unstandardise(pc[self.key], self.bounds, self.center)
        return pc
