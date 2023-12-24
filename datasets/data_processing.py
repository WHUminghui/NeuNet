# import pandas as pd
import numpy as np
from neuprint import Client
import os
from tqdm import tqdm
import glob
import shutil
from utils import count_files_numbers, count_file_numbers

def get_neuronID2type_dic():
    with open('../data/source_data/data_for_wangguojia/neuron-label.txt', 'r') as f:
        neuronID2type = {}
        for line in f.readlines():
            neuronID, neuron_type = line.strip().split('\t')[0], line.strip().split('\t')[1]
            neuronID2type[neuronID] = neuron_type
    return neuronID2type


def get_skeleton(save_path):
    TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6IjE0Mjg3NTYxNjhAcXEuY29tIiwibGV2ZWwiOiJub2F1dGgiLCJpbWFnZS11cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BSXRidm1ucHpZbGZCRTN0QlVyeERNY0ZONkVKeUhwVmJFVE15VDlZOURiZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTgzNzg1MDAxM30.qvtyuLhA6Qs6y0GQbBA4TRi8tJr1YmKqmk3xpqEnbxo"  # <--- Paste your token here
    # (or define NEUPRINT_APPLICATION CREDENTIALS in your environment

    c = Client('neuprint.janelia.org', 'hemibrain:v1.2.1', TOKEN)

    neuronID2type = get_neuronID2type_dic()
    for neuronID, neuron_type in tqdm.tqdm(neuronID2type.items()):
        if os.path.exists(f'{os.path.join(save_path, neuron_type)}/{neuronID}.csv'):
            continue
        neuron_skeleton = c.fetch_skeleton(neuronID, format='pandas')
        if not os.path.exists(os.path.join(save_path, neuron_type)):
            os.makedirs(os.path.join(save_path, neuron_type))
        neuron_skeleton.loc[:, ['x', 'y', 'z']].to_csv(f'{os.path.join(save_path, neuron_type)}/{neuronID}.csv', index=False)


def get_skeleton_rest(save_path):
    TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6IjE0Mjg3NTYxNjhAcXEuY29tIiwibGV2ZWwiOiJub2F1dGgiLCJpbWFnZS11cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BSXRidm1ucHpZbGZCRTN0QlVyeERNY0ZONkVKeUhwVmJFVE15VDlZOURiZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTgzNzg1MDAxM30.qvtyuLhA6Qs6y0GQbBA4TRi8tJr1YmKqmk3xpqEnbxo"  # <--- Paste your token here
    # (or define NEUPRINT_APPLICATION CREDENTIALS in your environment

    c = Client('neuprint.janelia.org', 'hemibrain:v1.2.1', TOKEN)
    rest = ['1405300458', '1467698336', '1655626125', '5813022707', '5813050779', '5813056516', '5813108230']
    neuronID2type = get_neuronID2type_dic()
    for neuronID in rest:
        neuron_type = neuronID2type[neuronID]
        if os.path.exists(f'{os.path.join(save_path, neuron_type)}/{neuronID}.csv'):
            continue
        neuron_skeleton = c.fetch_skeleton(neuronID, format='pandas')
        if not os.path.exists(os.path.join(save_path, neuron_type)):
            os.makedirs(os.path.join(save_path, neuron_type))
        neuron_skeleton.loc[:, ['x', 'y', 'z']].to_csv(f'{os.path.join(save_path, neuron_type)}/{neuronID}.csv', index=False)


def match(new_cate, old_cate):
    i, j = len(new_cate), len(old_cate)
    for index in range(i):
        if index == j:
            return False
        if new_cate[index]==old_cate[index]:
            continue
        else:
            return False
    return True


def coarsen_type(save_path):
    source_path = '../data/source_data/skeleton'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    new_categories = []
    categories = glob.glob(f'{source_path}/*')
    categories = sorted([x.split(os.sep)[-1] for x in categories])

    for category in categories:
        len_str = len(category) - 1
        while not category[len_str].isupper():
            len_str -= 1
        if len_str > 0:
            new_category = category[0:len_str + 1]
        else:
            new_category = category
        new_categories.append(new_category)
    new_categories = sorted(list(set(new_categories)))

    for i in range(len(new_categories)):
        new_cate = new_categories[i]
        new_categories[i] = [new_cate]
        for old_cate in categories:
            if match(new_cate, old_cate):
                new_categories[i].append(old_cate)


    with open(f'{save_path}/neurons_distrubution.txt', 'w') as file:
        old_typies = []
        for new_types in new_categories:
            file.write(f'{new_types[0]}: ')
            count = 0
            for old_type in set(new_types[1:]):
                if old_type in old_typies:
                    continue
                file.write(f'{old_type} ')
                count = count + len(glob.glob(f'{source_path}/{old_type}/*'))
                old_typies.append(old_type)
            file.write(f'{count}\n \n')

    old_typies = []
    for new_cateList in tqdm(new_categories):
        if not os.path.exists(f'{save_path}/{new_cateList[0]}'):
            os.makedirs(f'{save_path}/{new_cateList[0]}')
        for old_type in set(new_cateList[1:]):
            if old_type in old_typies:
                continue
            old_typies.append(old_type)
            for csv in glob.glob(f'{source_path}/{old_type}/*'):
                shutil.copyfile(f'{csv}', f'{save_path}/{new_cateList[0]}/{csv.split("/")[-1]}')


def get_skeleton_Dataset_raw(save_path, min):
    skeleton_path = '../data/source_data/coarsen_skeleton'
    dataset_path = f'{save_path}_more{min}'
    if not os.path.exists(dataset_path):
        shutil.copytree(skeleton_path, dataset_path)

    categories = glob.glob(f'{save_path}_more{min}/*')
    raw_path = f'{save_path}_more{min}/raw'
    lessmin_path = f'{save_path}_more{min}/NoMore{min}'
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
    if not os.path.exists(lessmin_path):
        os.makedirs(lessmin_path)

    with open(f'{dataset_path}/this_dataset_distubution.txt', 'w') as file:
        file.write('this dataset(raw) is:\n')
        typies_num = 0
        neuron_nums = []
        for category in categories:
            if len(glob.glob(f'{category}/*')) > min:
                typies_num += 1
                neuron_num = len(glob.glob(f'{category}/*'))
                neuron_nums.append(neuron_num)
                shutil.move(category, f'{raw_path}')
                file.write(f'{category.split(os.sep)[-1]}: {neuron_num}\n')
            else:
                shutil.move(category, f'{lessmin_path}')
        file.write(f'\nall nums of type(raw) is :{typies_num}. all neurons_num is {sum(neuron_nums)}')

    neurons_distubution = glob.glob(f'{lessmin_path}/neurons_distrubution.tx*')
    shutil.move(neurons_distubution[0], dataset_path)


def get_mesh(save_path):
    TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6IjE0Mjg3NTYxNjhAcXEuY29tIiwibGV2ZWwiOiJub2F1dGgiLCJpbWFnZS11cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BSXRidm1ucHpZbGZCRTN0QlVyeERNY0ZONkVKeUhwVmJFVE15VDlZOURiZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTgzNzg1MDAxM30.qvtyuLhA6Qs6y0GQbBA4TRi8tJr1YmKqmk3xpqEnbxo"  # <--- Paste your token here
    # (or define NEUPRINT_APPLICATION CREDENTIALS in your environment

    c = Client('neuprint.janelia.org', 'hemibrain:v1.2.1', TOKEN)

    neuronID2type = get_neuronID2type_dic()
    for neuronID, neuron_type in tqdm.tqdm(neuronID2type.items()):
        if os.path.exists(f'{os.path.join(save_path, neuron_type)}/{neuronID}.csv'):
            continue
        neuron_skeleton = c.fetch_roi_mesh(neuronID, format='pandas')
        if not os.path.exists(os.path.join(save_path, neuron_type)):
            os.makedirs(os.path.join(save_path, neuron_type))
        neuron_skeleton.loc[:, ['x', 'y', 'z']].to_csv(f'{os.path.join(save_path, neuron_type)}/{neuronID}.csv', index=False)



if __name__ == '__main__':
    # count_files_numbers('/data/users/minghuiliao/project/Dor/HemiBrain/data/source_data/coarsen_skeleton_more0/raw')
    get_neuronID2type_dic()

