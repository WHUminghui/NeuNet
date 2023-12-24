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
def nouse():
    sys.path.append(f'{os.path.dirname(__file__)}')
    wpath = os.getcwd()
    print(f'{os.path.dirname(__file__)}')
    print(wpath)
    print(os.path.exists("../data/source_data/data_for_wangguojia/edges.txt"))


def edges_txt2list(inpath):
    with open(inpath, 'r') as f:
        data = []
        for line in f:
            data_line = line.strip('\n').split(',')
            data.append([int(i) for i in data_line])
    return data

def get_neuronID():
    with open('/data/users/minghuiliao/project/Dor/HemiBrain/data/source_data/data_for_wangguojia/neuron2ID.txt', 'r') as file:
        data = {}
        for line in file:
            neuron, ID = [int(x) for x in line.strip().split(',')]
            data[neuron] = ID
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

def get_selected_neuronID(path):
    # path is ../../raw
    selected_neuronID = []
    neuron2ID = get_neuronID()
    categories_path = glob.glob(f'{path}/*')
    for category_path in categories_path:
        csvs_path = glob.glob(f'{category_path}/*')
        for csv_path in csvs_path:
            neuron = int(csv_path.strip().split(os.sep)[-1].split('.')[0])
            neuronID = neuron2ID[neuron]
            selected_neuronID.append(neuronID)
    selected_neuronID.sort()
    return selected_neuronID

def get_selected_neuronID_noNone(path):
    # path is ../../raw
    selected_neuronID = []
    neuron2ID = get_neuronID()
    categories_path = glob.glob(f'{path}/*')
    for category_path in categories_path:
        csvs_path = glob.glob(f'{category_path}/*')
        for csv_path in csvs_path:
            neuron = int(csv_path.strip().split(os.sep)[-1].split('.')[0])
            neuronID = neuron2ID[neuron]
            selected_neuronID.append(neuronID)
    selected_neuronID.sort()
    return selected_neuronID


def get_labels(Connect_raw_path):
    # path is '../../raw'
    density = Connect_raw_path.split(os.sep)[-2].split('e')[-1]
    skeleton_raw_path = f'/data/users/minghuiliao/project/Dor/HemiBrain/data/source_data/coarsen_skeleton_more{density}/raw'
    labels = []
    ID2labels = {}
    types = glob.glob(f'{skeleton_raw_path}/*')
    for type in types:
        type_name = type.strip().split(os.sep)[-1]
        labels.append(type_name)
        csvs = glob.glob(f'{type}/*csv')
        for csv in csvs:
            csv_name = csv.split('.')[0].split('/')[-1]
            ID2labels[int(csv_name)] = type_name
    return labels, ID2labels


def get_Connectome_raw(outdir, selected_neuronID, indir='/data/users/minghuiliao/project/Dor/HemiBrain/data/source_data/data_for_wangguojia'):
    if os.path.exists(osp.join(outdir, 'edge_index.npy')):
        return np.load(osp.join(outdir, 'edge_index.npy')),\
               np.load(osp.join(outdir, 'edge_attr.npy')),\
               np.load(osp.join(outdir, 'y.npy'))

    edges = edges_txt2list(osp.join(indir, 'edges.txt'))
    neuronID = edges_txt2list(osp.join(indir, 'neuron2ID.txt'))
    neuron2ID_dir = {}
    ID2neuron = []
    for i in range(len(neuronID)):
        key, value = neuronID[i][0], neuronID[i][1]
        neuron2ID_dir[key] = value
        ID2neuron.append(key)
    # with open(osp.join(indir, 'neuron-label.txt'), 'r') as f:
    #     ID2labels = {}
    #     labels = []
    #     for line in f:
    #         ID, label = line.strip('\n').split('\t')
    #         ID2labels[ID] = label
    #         labels.append(label)
    # labels = set(labels)
    labels, ID2labels = get_labels(outdir)
    labels2nums = {}
    for i, type in enumerate(labels):
        labels2nums[type] = i

    temp_edges = [[], []]
    temp_edge_attr = []
    print('creteating temp_edges...')
    for edge in tqdm(edges, position=0, maxinterval=len(edges[0])):
        temp_edges[0].append(neuron2ID_dir[edge[0]])
        temp_edges[1].append(neuron2ID_dir[edge[1]])
        temp_edge_attr.append(edge[2])
    print('creteated temp_edges')

    selected_neuronIDList = list(map(int, selected_neuronID))
    out_edges = [[], []]
    out_edge_attr = []
    print('selecting edge...')
    for i in tqdm(range(len(temp_edges[0])), position=0):
        if temp_edges[0][i] in selected_neuronID and temp_edges[1][i] in selected_neuronID:
            out_edges[0].append(selected_neuronIDList.index(temp_edges[0][i]))
            out_edges[1].append(selected_neuronIDList.index(temp_edges[1][i]))
            out_edge_attr.append(temp_edge_attr[i])
    print('selected edge!')

    out_label = []
    for id in selected_neuronID:
        out_label.append(labels2nums[ID2labels[ID2neuron[int(id)]]])
    np.save(osp.join(outdir, 'edge_index.npy'), np.array(out_edges))
    np.save(osp.join(outdir, 'edge_attr.npy'), np.array(out_edge_attr))
    np.save(osp.join(outdir, 'y.npy'), np.array(out_label))

    return np.array(out_edges), np.array(out_edge_attr), np.array(out_label)


def get_Connectome_raw_WithoutNone(outdir, selected_neuronID, indir='/data/users/minghuiliao/project/Dor/HemiBrain/data/source_data/data_for_wangguojia'):
    if os.path.exists(osp.join(outdir, 'edge_index.npy')):
        return np.load(osp.join(outdir, 'edge_index.npy')),\
               np.load(osp.join(outdir, 'edge_attr.npy')),\
               np.load(osp.join(outdir, 'y.npy'))

    edges = edges_txt2list(osp.join(indir, 'edges.txt'))
    neuronID = edges_txt2list(osp.join(indir, 'neuron2ID.txt'))
    neuron2ID_dir = {}
    ID2neuron = []
    for i in range(len(neuronID)):
        key, value = neuronID[i][0], neuronID[i][1]
        neuron2ID_dir[key] = value
        ID2neuron.append(key)
    # with open(osp.join(indir, 'neuron-label.txt'), 'r') as f:
    #     ID2labels = {}
    #     labels = []
    #     for line in f:
    #         ID, label = line.strip('\n').split('\t')
    #         ID2labels[ID] = label
    #         labels.append(label)
    # labels = set(labels)
    labels, ID2labels = get_labels(outdir)
    labels2nums = {}
    for i, type in enumerate(labels):
        labels2nums[type] = i

    temp_edges = [[], []]
    temp_edge_attr = []
    print('creteating temp_edges...')
    for edge in tqdm(edges, position=0, maxinterval=len(edges[0])):
        temp_edges[0].append(neuron2ID_dir[edge[0]])
        temp_edges[1].append(neuron2ID_dir[edge[1]])
        temp_edge_attr.append(edge[2])
    print('creteated temp_edges')

    selected_neuronIDList = list(map(int, selected_neuronID.tolist()))
    out_edges = [[], []]
    out_edge_attr = []
    print('selecting edge...')
    for i in tqdm(range(len(temp_edges[0])), position=0):
        if temp_edges[0][i] in selected_neuronID and temp_edges[1][i] in selected_neuronID:
            out_edges[0].append(selected_neuronIDList.index(temp_edges[0][i]))
            out_edges[1].append(selected_neuronIDList.index(temp_edges[1][i]))
            out_edge_attr.append(temp_edge_attr[i])
    print('selected edge!')

    out_label = []
    for id in selected_neuronID:
        out_label.append(labels2nums[ID2labels[ID2neuron[int(id)]]])
    np.save(osp.join(outdir, 'edge_index.npy'), np.array(out_edges))
    np.save(osp.join(outdir, 'edge_attr.npy'), np.array(out_edge_attr))
    np.save(osp.join(outdir, 'y.npy'), np.array(out_label))

    return np.array(out_edges), np.array(out_edge_attr), np.array(out_label)


class HemiConnectome(InMemoryDataset):
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
        density = int(self.raw_dir.split(os.sep)[-2].split('e')[-1])
        skeleton_raw_path = f'/data/users/minghuiliao/project/Dor/HemiBrain/data/source_data/coarsen_skeleton_more{density}/raw'
        selected_ID = get_selected_neuronID(skeleton_raw_path)
        edges_index, edges_attr, y = get_Connectome_raw(outdir=self.raw_dir, selected_neuronID=selected_ID)
        self.selected_ID = selected_ID

        index = torch.randperm(len(y)).tolist()
        train_index = index[:len(y)//10*8]
        test_index = index[len(y)//10*8:len(y)//10*9]
        val_index = index[len(y)//10*9:]

        # train_index = torch.arange(len(y)//10*8, dtype=torch.long)
        # test_index = torch.arange(len(y)//10*8, len(y)//10*9, dtype=torch.long)
        # val_index = torch.arange(len(y) // 10 * 9, len(y), dtype=torch.long)

        train_mask = torch.zeros((len(y), ), dtype=torch.bool)
        val_mask = torch.zeros((len(y),), dtype=torch.bool)
        test_mask = torch.zeros((len(y), ), dtype=torch.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        data = Data(edge_index=torch.tensor(edges_index), edge_attr=torch.tensor(edges_attr), y=torch.tensor(y))
        data.selected_ID = selected_ID
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'


class HemiConnectome_WithoutNone(InMemoryDataset):
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
        meta_model = PointNetSkeleton(1)
        selected_ID, hidden_features, max_data_acc = meta_model.pre_train(self.raw_dir, 13)
        selected_ID = selected_ID.tolist()
        edges_index, edges_attr, y = get_Connectome_raw_WithoutNone(outdir=self.raw_dir, selected_neuronID=selected_ID)
        train_index = torch.arange(len(y)//10*8, dtype=torch.long)
        test_index = torch.arange(len(y)//10*8, len(y), dtype=torch.long)
        None_index = get_None_neuronsID()
        # None_indexID = []
        # for i in None_index:
        #     if not i in selected_ID:
        #         continue
        #     indexID = selected_ID.index(i)
        #     None_indexID.append(indexID)
        train_mask = torch.zeros((len(y), ), dtype=torch.bool)
        test_mask = torch.zeros((len(y), ), dtype=torch.bool)
        train_mask[train_index] = True
        test_mask[test_index] = True

        for i in None_index:
            if i in selected_ID:
                train_mask[selected_ID.index(i)] = False
                test_mask[selected_ID.index(i)] = False

        data = Data(x=torch.from_numpy(np.float32(hidden_features)), edge_index=torch.tensor(edges_index),\
                    edge_attr=torch.tensor(edges_attr), y=torch.tensor(y))
        data.train_mask = train_mask
        data.test_mask = test_mask
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'


if __name__ == '__main__':
    for density in [5, 20, 50, 100, 200, 500]:
        path_Skeleton = f'../data/source_data/HemiConnectome{density}'
        transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
        dataset = HemiConnectome(path_Skeleton, transform=transform)

