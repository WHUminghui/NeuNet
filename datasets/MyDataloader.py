from collections.abc import Mapping, Sequence
from typing import List, Optional, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data import InMemoryDataset


class Collater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # Deprecated...
        return self(batch)


class MyDataLoader():
    def __init__(self, dataset: InMemoryDataset, batch_size, edge_index, edge_attr):
        assert batch_size > 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def __iter__(self):
        self.step = 0
        perm = torch.randperm(len(self.data))
        self.batch_index = torch.split(perm, self.batch_size)
        return self

    def __next__(self):
        if self.step >= len(self.batch_index):
            raise StopIteration
        else:
            if len(self.batch_index) >= len(self.data):
                return self.data
            batch = [self.data[i] for i in self.batch_index[self.step].tolist()]
            self.step += 1
        return batch, self.edge_index, self.edge_attr




if __name__ == '__main__':
    pass