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

def edges_txt2list(inpath):
    with open(inpath, 'r') as f:
        data = []
        for line in f:
            data_line = line.strip('\n').split(',')
            data.append([int(i) for i in data_line])
    return data


def get_None_neuronsID():
    None_path = 'data/source_data/coarsen_skeleton/None'
    csvs = glob.glob(f'{None_path}/*')
    None_neuronsIDs = []
    for csv in csvs:
        None_neuronsID = int(csv.strip().split('.')[0].split(os.sep)[-1])
        None_neuronsIDs.append(None_neuronsID)
    indir = '/data/users/minghuiliao/project/Dor/HemiBrain/data/source_data/data_for_wangguojia'
    neuronID = edges_txt2list(osp.join(indir, 'neuron2ID.txt'))
    neuron2ID_dir = {}
    for i in range(len(neuronID)):
        key, value = int(neuronID[i][0]), int(neuronID[i][1])
        neuron2ID_dir[key] = value
    None_IDs = []
    for neuron in None_neuronsIDs:
        None_IDs.append(neuron2ID_dir[neuron])

    return None_IDs


def read_csv(path, ID, neuron):
    pos = pd.read_csv(path)
    pos_list = pos.values.tolist()
    pos_tensor = torch.tensor(pos_list).squeeze()
    data = Data(pos=pos_tensor, ID=ID, neuron=neuron)
    return data


def get_neuronID():
    with open('/data/users/minghuiliao/project/Dor/HemiBrain/data/source_data/data_for_wangguojia/neuron2ID.txt', 'r') as file:
        data = {}
        for line in file:
            neuron, ID = [int(x) for x in line.strip().split(',')]
            data[neuron] = ID
    return data


class HemiSkeleton(InMemoryDataset):
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
        categories.pop(categories.index('None'))
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
                data = read_csv(path, ID, neuron)
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


class HemiSkeleton_withNone(InMemoryDataset):
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
        # categories.pop(categories.index('None'))
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
                data = read_csv(path, ID)
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


class HemiSkeletonMask(InMemoryDataset):
    def __init__(self, root, train_ratio=0.8, test_ratio=0.2, transform=None, pre_transform=None, pre_filter=None):
        self.train_ratio, self.test_ratio = train_ratio, test_ratio
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

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
        return 'data.pt'

    def download(self):
        pass

    def process(self):
            torch.save(self.collate([self.process_set()]), self.processed_paths[0])

    def process_set(self):
        # to create BrianData using raw file and Data class
        print("HemiSkeletonMask Dataset process_set() is working")
        categories = glob.glob(f'{self.raw_dir}/*')
        categories = sorted([x.split(os.sep)[-1] for x in categories])
        neuron2ID = get_neuronID()
        data_list = []
        selected_ID = []
        for target, category in enumerate(tqdm(categories, position=0)):
            folder = osp.join(self.raw_dir, category)
            paths = glob.glob(f'{folder}/*')
            for path in paths:
                neuron = int(path.split(os.sep)[-1].split('.')[0])
                ID = neuron2ID[neuron]
                selected_ID.append(ID)
                data = read_csv(path, ID)
                data.y = torch.tensor([target])
                data_list.append(data)
        selected_ID.sort()
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        train_index = torch.arange(len(data_list)//10*8, dtype=torch.long)
        test_index = torch.arange(len(data_list)//10*8, len(data_list), dtype=torch.long)
        None_index = get_None_neuronsID()
        train_mask = torch.zeros((len(data_list), ), dtype=torch.bool)
        test_mask = torch.zeros((len(data_list), ), dtype=torch.bool)
        train_mask[train_index] = True
        test_mask[test_index] = True
        for i in None_index:
            if i in selected_ID:
                train_mask[selected_ID.index(i)] = False
                test_mask[selected_ID.index(i)] = False
        np.save(f'{self.root}/train_mask.npy', np.array(train_mask))
        np.save(f'{self.root}/test_mask.npy', np.array(test_mask))
        np.save(f'{self.root}/num_class.npy', np.array(len(categories) - 1))

        return data_list

    def get_mask(self):
        train_mask = np.load(f'{self.root}/train_mask.npy')
        test_mask = np.load(f'{self.root}/test_mask.npy')
        num_class = np.save(f'{self.root}/num_class.npy')
        return train_mask, test_mask, num_class

    def __repr__(self) -> str:
        # print the BrainData class info for print(BrainData)
        return f'{self.__class__.__name__}({len(self)})'


if __name__ == '__main__':
    for density in [0, 3, 5, 20, 50, 100, 500]:
        path = f'../data/source_data/coarsen_skeleton_more{density}'
        pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)
        data_dataset = HemiSkeleton(path, 'data', transform=transform, pre_transform=pre_transform)
        print(density)




























