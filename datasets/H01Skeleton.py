import glob
import os
import shutil
import os.path as osp
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data import Data
from typing import Union, List, Tuple
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius
from tqdm import tqdm
import numpy as np
import torch_geometric.transforms as T
import shutil


def get_raw(raw_path):
    path = '../data/h01_c3/skeletons'
    for i in tqdm(range(10), position=0):
        skes = np.load(f'{path}/skes{i}.npy')
        indexs = np.load(f'{path}/index{i}.npy')
        ids = np.load(f'{path}/../neurons_id/ids_{i}.npy')
        for index, id in zip(indexs, ids):
            if index == 0:
                continue
            ske = skes[:index]
            skes = skes[index:]
            np.save(f'{raw_path}/{id}.npy', ske)

def get_label():
    labels = []
    with open('../data/h01_c3/label/label.txt', 'r') as f:
        for line in f.readlines():
            labels.append(line)
    l1 = [int(i) for i in labels[1].split(',')]
    l2 = [int(i) for i in labels[3].split(',')]
    l3 = [int(i) for i in labels[5].split(',')]
    l4 = [int(i) for i in labels[7].split(',')]
    l5 = [int(i) for i in labels[9].split(',')]
    l6 = [int(i) for i in labels[11].split(',')]
    return l1, l2, l3, l4, l5, l6

def move_raw_file():
    l1, l2, l3, l4, l5, l6 = get_label()
    path = '../data/h01_c3/H01_Skeletons/raw'
    for ids, label in tqdm(zip([l1, l2, l3, l4, l5, l6], ['l1', 'l2', 'l3', 'l4', 'l5', 'l6'])):
        if not os.path.exists(f'{path}/{label}/'):
            os.makedirs(f'{path}/{label}/')
        for id in ids:
            try:
                shutil.move(f'{path}/{id}.npy', f'{path}/{label}/')
            except FileNotFoundError:
                continue

def get_neuronID():
    neuron2id = {}
    skeleton_raw_path = '/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Skeletons/raw'
    categories = glob.glob(f'{skeleton_raw_path}/*')
    categories = sorted([x.split(os.sep)[-1] for x in categories])
    id = 0
    for category in categories:
        neurons = sorted([int(x.split(os.sep)[-1].split('.')[0]) for x in glob.glob(f'{skeleton_raw_path}/{category}/*')])
        for neuron in neurons:
            neuron2id[neuron] = id
            id += 1
    return neuron2id

def read_npy(path, ID):
    pos = np.load(path)
    # pos_list = pos.values.tolist()
    pos_tensor = torch.tensor(pos) # .squeeze()
    data = Data(pos=pos_tensor, ID=ID)
    return data

class H01Skeleton(InMemoryDataset):
    def __init__(self, root, load_data, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1,
                 transform=None, pre_transform=None, pre_filter=None):
        # assert train in [True, False]
        self.train_ratio, self.test_ratio, self.val_ratio, = train_ratio, test_ratio, val_ratio
        super().__init__(root, transform, pre_transform, pre_filter)
        assert load_data in ['data', 'train', 'test', 'val']
        if load_data == 'data':
            path = self.processed_paths[0]
        elif load_data == 'train':
            path = self.processed_paths[1]
        elif load_data == 'test':
            path = self.processed_paths[2]
        elif load_data == 'val':
            path = self.processed_paths[3]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        # return the all file names in th file: raw
        # neurons_label = pd.read_csv('Data/data_for_wangguojia/neuron-label.txt', sep='\t', header=None)
        # lables = sorted(list(set(neurons_label.iloc[:, 1].tolist())))
        # return lables

        paths = glob.glob(f'{self.raw_dir}/*')
        types = []
        for path in paths:
            type = path.split('/')[-1]
            types.append(type)
        return types

    @property
    def processed_file_names(self):
        # return the all file names in th file: processed
        return ['data.pt', 'train.pt', 'test.pt', 'val.pt']

    def download(self):
        pass

    def process(self):
        data, train, test, val = self.process_set()
        torch.save(data, self.processed_paths[0])
        torch.save(train, self.processed_paths[1])
        torch.save(test, self.processed_paths[2])
        torch.save(val, self.processed_paths[3])

    def process_set(self):
        # to create BrianData using raw file and Data class
        categories = glob.glob(f'{self.raw_dir}/*')
        categories = sorted([x.split(os.sep)[-1] for x in categories])
        ###
        categories.pop(categories.index('unlabel'))
        ###
        neuron2ID = get_neuronID()
        data_list = []
        print("Creating SkeletonData...")
        for target, category in enumerate(tqdm(categories, position=0)):
            folder = osp.join(self.raw_dir, category)
            paths = glob.glob(f'{folder}/*')
            for path in paths:
                neuron = int(path.split(os.sep)[-1].split('.')[0])
                ID = neuron2ID[neuron]
                data = read_npy(path, ID)
                data.y = torch.tensor([target])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        index = torch.randperm(len(data_list)).tolist()
        train_index = index[:int(self.train_ratio * len(data_list))]
        test_index = index[int(self.train_ratio * len(data_list)):int(self.test_ratio * len(data_list)) + int(
            self.train_ratio * len(data_list))]
        val_index = index[int(self.test_ratio * len(data_list)) + int(self.train_ratio * len(data_list)):]

        train_list = [data_list[i] for i in train_index]
        test_list = [data_list[i] for i in test_index]
        val_list = [data_list[i] for i in val_index]

        return self.collate(data_list), self.collate(train_list), self.collate(test_list), self.collate(val_list)

    def __repr__(self) -> str:
        # print the BrainData class info for print(BrainData)
        return f'{self.__class__.__name__}({len(self)})'



if __name__ == '__main__':
    # get_raw('../data/h01_c3/H01_Skeletons/raw')
    # move_raw_file()
    pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)
    data = H01Skeleton('/data/users/minghuiliao/project/Dor/HemiBrain/data/h01_c3/H01_Skeletons', 'data', transform=transform, pre_transform=pre_transform)
    end = 0




