from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP, StateTSPDist
from utils.beam_search import beam_search
import torch.nn.functional as F


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


# class TSPCoordDist(TSP):

#     NAME = 'tsp_coorddist'

#     @staticmethod
#     def get_costs(dataset, pi):
#         # Check that tours are valid, i.e. contain 0 to n -1
#         assert (
#             torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
#             pi.data.sort(1)[0]
#         ).all(), "Invalid tour"

#         # Gather dataset in order of tour
#         d = dataset[:,:,:2].gather(1, pi.unsqueeze(-1).expand_as(dataset[:,:,:2]))

#         # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
#         return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

#     @staticmethod
#     def make_dataset(*args, **kwargs):
#         return TSPCoordDistDataset(*args, **kwargs)

#     @staticmethod
#     def make_state(*args, **kwargs):
#         kwargs['dist'] = args[0][:,:,2:] # (batch_size, graph_size, num_dist)
#         return StateTSP.initialize(args[0][:,:,:2], # (batch_size, graph_size, 2)
#                                    *args[1:], **kwargs)


# # def pdist(x):
# #     xx = F.pdist(x)
# #     m = torch.zeros((x.shape[0],x.shape[0]))
# #     triu_indices = torch.triu_indices(row=x.shape[0], col=x.shape[0], offset=1)
# #     m[triu_indices[0], triu_indices[1]] = xx
# #     m[triu_indices[1], triu_indices[0]] = xx
# #     return m


# class TSPCoordDistDataset(Dataset):
#     def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
#         super(TSPCoordDistDataset, self).__init__()

#         self.data_set = []
#         if filename is not None:
#             assert os.path.splitext(filename)[1] == '.pkl'

#             with open(filename, 'rb') as f:
#                 data = pickle.load(f)
#                 self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
#         else:
#             # Sample points randomly in [0, 1] square
#             self.data = torch.stack([torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)])
        
#         #self.distances = [pdist(d) for d in self.data]
#         self.distances = (self.data[:, :, None, :] - self.data[:, None, :, :]).norm(p=2, dim=-1)

#         self.size = len(self.data)

#     def __len__(self):
#         return self.size

#     def __getitem__(self, idx):
#         return torch.cat([self.data[idx],self.distances[idx]],dim=1)



class TSPDist(TSP):

    NAME = 'tsp_dist'

    @staticmethod
    def get_costs(dataset, pi):
        # pi: matrix (batch_size, graph_size)

        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        a = torch.arange(pi.shape[1])
        idx = torch.stack((a, a.roll(-1,0)))  # neighbor cities
        return dataset[:,idx[0],idx[1]].sum(1), None

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSPDist.initialize(*args, **kwargs)

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDistDataset(*args, **kwargs)


class TSPDistDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDistDataset, self).__init__()

        self.graph_size = size
        self.size = num_samples

        if filename is not None:
            raise NotImplementedError
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample distances in [0, 1]
            self.seeds = torch.zeros((num_samples,), dtype=torch.int64)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        self.seeds[idx] = torch.initial_seed()
        r = torch.FloatTensor(self.graph_size, self.graph_size).uniform_(0, 1).triu(1)
        return r + r.t()