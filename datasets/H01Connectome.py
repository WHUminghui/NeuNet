import glob
import os
import os.path as osp
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data import Data
from typing import Union, List, Tuple
import torch
import numpy as np
from datasets.HemiSkeleton import HemiSkeleton
from model.model import PointNetSkeleton
import sys
from tqdm import tqdm
import torch_geometric.transforms as T

def get_label():
    skeleton_raw_path = '/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Skeletons/raw'
    if os.path.exists('/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Connectome/raw/y.npy'):
        return np.load('/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Connectome/raw/y.npy')
    categories = glob.glob(f'{skeleton_raw_path}/*')
    categories = sorted([x.split(os.sep)[-1] for x in categories])
    label = []
    for y, categoroy in enumerate(categories):
        for neuron in glob.glob(f'{skeleton_raw_path}/{categoroy}/*'):
            label.append(y)
    np.save('/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Connectome/raw/y.npy', label)
    return np.array(label)


def get_neuronID():
    neuron2id = {}
    neuronID = []
    skeleton_raw_path = '/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Skeletons/raw'
    categories = glob.glob(f'{skeleton_raw_path}/*')
    categories = sorted([x.split(os.sep)[-1] for x in categories])
    id = 0
    for category in categories:
        neurons = sorted([int(x.split(os.sep)[-1].split('.')[0]) for x in glob.glob(f'{skeleton_raw_path}/{category}/*')])
        for neuron in neurons:
            neuron2id[neuron] = id
            neuronID.append(neuron)
            id += 1
    return neuron2id, neuronID


def get_edge_raw():
    path = '/data/users/minghuiliao/project/Dor/HemiBrain/data'
    if os.path.exists(f'{path}/h01_c3/H01_Connectome/raw/edge_index_05.npy'):
        return
    _, neurons = get_neuronID()
    neurons = np.array(neurons)
    edge_paths = glob.glob(f'{path}/h01_c3/synapses_edge/*')
    for edge_path in tqdm(edge_paths, position=0):
        name = edge_path.split(os.sep)[-1].split('.')[0].split('_')[-1]
        edge_all = np.load(f'{edge_path}', allow_pickle=True).astype(float)
        interact0 = np.intersect1d(neurons, edge_all[0])
        interact1 = np.intersect1d(neurons, edge_all[1])
        mask0 = np.zeros(len(edge_all[0]), dtype=np.bool_)
        mask1 = np.zeros(len(edge_all[1]), dtype=np.bool_)
        index0 = np.where(edge_all[0] == interact0[:, None])[-1]
        index1 = np.where(edge_all[1] == interact1[:, None])[-1]
        mask0[index0] = 1
        mask1[index1] = 1
        mask = mask0 & mask1
        edge_index = edge_all[:, mask]
        np.save(f'{path}/h01_c3/H01_Connectome/raw/edge_index_{name}.npy', edge_index)
def get_edge_attr_raw():
    get_edge_raw()
    if os.path.exists(f'/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Connectome/raw/edge_index.npy'):
        return  np.load('/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Connectome/raw/edge_index.npy'),\
                np.load('/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Connectome/raw/edge_attr.npy')
    edge = glob.glob(f'/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Connectome/raw/*')
    edge_index = np.load(f'{edge[0]}').astype(int)
    for fracton_path in edge[1:]:
        if fracton_path == '/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Connectome/raw/edge_attr.npy':
            continue
        edge_index_temp = np.load(f'{fracton_path}').astype(int)
        edge_index = np.hstack((edge_index, edge_index_temp))
    edge_attr = {}
    nocover = []
    for i in range(len(edge_index[0])):
        if not f'{edge_index[0][i]}_{edge_index[1][i]}' in edge_attr:
            nocover.append(i)
        edge_attr[f'{edge_index[0][i]}_{edge_index[1][i]}'] = edge_attr.get(f'{edge_index[0][i]}_{edge_index[1][i]}', 0) + 1
    mask = np.zeros(len(edge_index[0]), dtype=np.bool_)
    mask[np.array(nocover)] = 1
    out_edge0 = edge_index[0][mask]
    out_edge1 = edge_index[1][mask]
    out_edge = np.vstack((out_edge0, out_edge1))
    attr = []
    for value in edge_attr.values():
        attr.append(value)
    attr = np.array(attr)
    np.save('/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Connectome/raw/edge_index.npy', out_edge)
    np.save('/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Connectome/raw/edge_attr.npy', attr)
    return np.load('/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Connectome/raw/edge_index.npy'), \
           np.load('/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Connectome/raw/edge_attr.npy')


def get_raw():
    edge_index, edge_attr = get_edge_attr_raw()
    neuron2id, _ = get_neuronID()
    y = get_label()
    for row in range(len(edge_index)):
        for col in range(len(edge_index[0])):
            edge_index[row][col] = neuron2id[edge_index[row][col]]
    return edge_index, edge_attr, y


class H01Connectome(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['edge_index.npy', 'edge_attr,npy', 'y.npy']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        torch.save(self.collate([self.process_set()]), self.processed_paths[0])

    def process_set(self):
        edges_index, edges_attr, y = get_raw()
        # - nums(unlabel)
        index = torch.randperm(len(y)-1096).tolist()
        train_index = index[:len(y)//10*8]
        test_index = index[len(y)//10*8:len(y)//10*9]
        val_index = index[len(y)//10*9:]
        train_mask = torch.zeros((len(y), ), dtype=torch.bool)
        val_mask = torch.zeros((len(y),), dtype=torch.bool)
        test_mask = torch.zeros((len(y), ), dtype=torch.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

        data = Data(edge_index=torch.tensor(edges_index), edge_attr=torch.tensor(edges_attr), y=torch.tensor(y))
        selected_ID = range(0, len(y))
        data.selected_ID = selected_ID
        data.num_nodes = len(y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'


def nouse():
    edge = np.load('../data/h01_c3/synapses_edge/synapses_edge_0e.npy', allow_pickle=True)
    end = 0


if __name__ == '__main__':
    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    path = '../data/h01_c3/H01_Connectome'
    dataset = H01Connectome(path, transform=transform)
    end = 0