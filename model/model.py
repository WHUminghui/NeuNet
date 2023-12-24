import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from datasets.HemiSkeleton import HemiSkeleton
from tqdm import tqdm
import numpy as np
import os
from torch import nn
from torch.nn import BatchNorm1d

class GCNII_withoutLinear(torch.nn.Module):
    def __init__(self, dataset_num_features, dataset_num_classes, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset_num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset_num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers - 1):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
        self.lastLinear = Linear(hidden_channels, dataset_num_classes)
        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lastLinear(x)

        return x.log_softmax(dim=-1)


class GCNII_x2learning(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, dataset_num_classes, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset_num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset_num_classes))
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, adj_t):
        x = self.x # is copy? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x.log_softmax(dim=-1)


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNetSkeleton(torch.nn.Module):
    def __init__(self, num_class):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
        self.num_class = num_class
        self.mlp = MLP([1024, (1024-self.num_class)//2 + self.num_class, self.num_class], dropout=0.5)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        x_out = self.mlp(x)
        y = x_out.log_softmax(dim=-1)
        return y

    def pre_train(self, raw_dir, max_epoch):
        density = raw_dir.split(os.sep)[-2].split('e')[-1]
        path =f'data/source_data/coarsen_skeleton_more{density}'
        if os.path.exists(f'{path}/pre_train/selected_ID.npy'):
            max_data_acc = np.load(f'{path}/pre_train/max_data_acc.npy')
            print(f'the result of pre_train(pre_train) has been saved!, the max_data_acc is {max_data_acc}')
            selected_ID = np.load(f'{path}/pre_train/selected_ID.npy')
            hidden_features= np.load(f'{path}/pre_train/hidden_features.npy')
        else:
            # density = raw_dir.split(os.sep)[-2].split('e')[-1]
            # path = f'data/source_data/coarsen_skeleton_more{density}'
            print('PointNetSkeleton(pre_train) is pre_trainning...')
            pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)
            data_dataset = HemiSkeleton(path, 'test', transform=transform, pre_transform=pre_transform)
            data_loader = DataLoader(data_dataset, batch_size=32, shuffle=True, num_workers=6, drop_last=True)

            device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
            model = PointNetSkeleton(max(data_dataset.data.y.tolist()) + 1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            max_data_acc = 0
            for epoch in tqdm(range(max_epoch), colour='blue'):
                # hidden_features = train(epoch, max_epoch - 1)

                hidden_features = []
                ##### train() #####
                model.train()
                if epoch == max_epoch - 1:
                    for data in data_loader:
                        data = data.to(device)
                        optimizer.zero_grad()
                        ypre, hidden_feature = model(data)
                        loss = F.nll_loss(ypre, data.y)
                        loss.backward()
                        optimizer.step()
                        hidden_features += hidden_feature
                    hidden_features = np.array(hidden_features, dtype=float)
                    hidden_features = hidden_features[np.lexsort(hidden_features[:, ::-1].T)]
                    selected_ID = hidden_features[:, 0]
                    hidden_features = hidden_features[:, 1:]

                    ##### test()  ####
                    model.eval()
                    correct = 0
                    for data in data_loader:
                        data = data.to(device)
                        with torch.no_grad():
                            pred, _ = model(data)
                            pred = pred.max(1)[1]
                        correct += pred.eq(data.y).sum().item()
                    data_acc = correct/len(selected_ID)
                    max_data_acc = max(max_data_acc, data_acc)
                    print(f'Last: Epoch: {epoch:03d}, Train: {data_acc:.4f}, max_data_acc: {max_data_acc:.4f}')
                    #####

                else:
                    for data in data_loader:
                        data = data.to(device)
                        optimizer.zero_grad()
                        ypre, hidden_feature = model(data)
                        loss = F.nll_loss(ypre, data.y)
                        loss.backward()
                        optimizer.step()
                        hidden_features += hidden_feature

                    ##### test()  ####
                    model.eval()
                    correct = 0
                    for data in data_loader:
                        data = data.to(device)
                        with torch.no_grad():
                            pred, _ = model(data)
                            pred = pred.max(1)[1]
                        correct += pred.eq(data.y).sum().item()
                    data_acc = correct/len(hidden_features)
                    max_data_acc = max(max_data_acc, data_acc)
                    print(f'Epoch: {epoch:03d}, Train: {data_acc:.4f}, max_data_acc: {max_data_acc:.4f}')

            print(f'pre_train has been done! the max_data_acc is {max_data_acc}')
            os.makedirs(f'{path}/pre_train')
            np.save(f'{path}/pre_train/selected_ID.npy', selected_ID)
            np.save(f'{path}/pre_train/hidden_features.npy', hidden_features)
            np.save(f'{path}/pre_train/max_data_acc.npy', np.array(max_data_acc))
        return selected_ID, hidden_features, max_data_acc


class GCNII(torch.nn.Module):
    def __init__(self, dataset_num_features, dataset_num_classes, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset_num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset_num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


class PointNetSkeleton_Concat(torch.nn.Module):
    def __init__(self, dataset_num_features, represent_features, num_class, num_PointNet2Layer):
        super().__init__()
        assert represent_features <= dataset_num_features
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 256]))
        self.convs = torch.nn.ModuleList()
        for layer in range(num_PointNet2Layer - 2):
            self.convs.append(SAModule(0.25, 0.4, MLP([256 + 3, 256, 256, 256])))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, dataset_num_features]))
        self.represent_features = represent_features
        self.mlp = MLP([dataset_num_features, (dataset_num_features - self.represent_features) // 3 + self.represent_features,\
                        2*(dataset_num_features - self.represent_features) // 3 + self.represent_features,\
                        self.represent_features], dropout=0.5)
        self.outL = Linear(represent_features, num_class)
    def forward(self, data):
        sa_out = (data.x, data.pos, data.batch)
        sa_out = self.sa1_module(*sa_out)
        for conv in self.convs:
            sa_out = conv(*sa_out)
        sa_out = self.sa3_module(*sa_out)
        x, pos, batch = sa_out
        x = self.mlp(x)

        y = self.outL(x)
        # y = y.log_softmax(dim=-1)
        return x, y


class GCNII_Concat(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_features, num_calss, represent_features=512,\
                 hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset_num_features, hidden_channels))
        self.mlp = MLP([hidden_channels, (represent_features - hidden_channels) // 3 + hidden_channels,\
                        2*(represent_features - hidden_channels) // 3 + hidden_channels,\
                        represent_features], dropout=0.5)
        self.x = nn.Parameter(torch.Tensor(num_nodes, dataset_num_features))
        torch.nn.init.xavier_uniform(self.x, gain=1)
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer + 1, shared_weights, normalize=False))
        self.dropout = dropout
        self.outL = Linear(represent_features, num_calss)

    def forward(self, adj_t):
        x = self.x # is deep copy? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()
        x = self.mlp(x)

        y = self.outL(x)
        # y = y.log_softmax(dim=-1)

        # x = F.dropout(x, self.dropout, training=self.training)
        # x_temp = x
        # x = self.lins[1](x)
        # y = x.log_softmax(dim=-1)
        return x, y



class BrainConnetome_concat_X(torch.nn.Module):
    # concat x1, x2
    def __init__(self, num_nodes, dataset_num_classes, dataset_num_features=1024, represent_features=512,\
                 hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0):
        super().__init__()
        self.represent_features = represent_features
        self.PointNetSkeleton_Concat = PointNetSkeleton_Concat(dataset_num_features, self.represent_features, num_class=dataset_num_classes)
        self.GCNII_forConcat = GCNII_Concat(num_nodes, dataset_num_features, dataset_num_classes, self.represent_features,\
                                            hidden_channels, num_layers, alpha, theta, shared_weights, dropout)

        self.mlp = MLP([2*self.represent_features, (2*self.represent_features - dataset_num_classes) // 3 + dataset_num_classes,\
                        2*(2*self.represent_features - dataset_num_classes) // 3 + dataset_num_classes,\
                        dataset_num_classes], dropout=0.5)


    def forward(self, device, data_Skeleton, adj_t_Connnecton, selected_ID):
        x_Connectome, _ = self.GCNII_forConcat(adj_t_Connnecton)
        x_Skeleton, _ = self.PointNetSkeleton_Concat(data_Skeleton)
        neuronIDs = data_Skeleton.ID.tolist()

        # x_selected_Connectome = x_Connectome[selected_ID.index(neuronIDs[0])]
        # x_selected_Connectome = x_selected_Connectome.view(1, -1)
        # for i in range(len(neuronIDs) - 1):
        #     i += 1
        #     index = selected_ID.index(neuronIDs[i])
        #     b = x_Connectome[index].view(1, -1)
        #     x_selected_Connectome = torch.cat((x_selected_Connectome, b), 0)

        index_Connectome = []
        for i in range(len(neuronIDs)):
            index_Connectome.append(selected_ID.index(neuronIDs[i]))
        x_selected_Connectome = x_Connectome[index_Connectome]


        x_total = torch.cat((x_selected_Connectome, x_Skeleton), 1)
        x_total = self.mlp(x_total)
        x = x_total.log_softmax(dim=-1)
        return x
class BrainConnetome_sum_Y(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_classes, dataset_num_features=1024, represent_features=512,\
                 hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0):
        super().__init__()
        self.represent_features = represent_features
        self.PointNetSkeleton_Concat = PointNetSkeleton_Concat(dataset_num_features, self.represent_features, num_class=dataset_num_classes)
        self.GCNII_forConcat = GCNII_Concat(num_nodes, dataset_num_features, dataset_num_classes, self.represent_features,\
                                            hidden_channels, num_layers, alpha, theta, shared_weights, dropout)

        self.mlp = MLP([2*self.represent_features, (2*self.represent_features - dataset_num_classes) // 3 + dataset_num_classes,\
                        2*(2*self.represent_features - dataset_num_classes) // 3 + dataset_num_classes,\
                        dataset_num_classes], dropout=0.5)

    def forward(self, device, data_Skeleton, adj_t_Connnecton, selected_ID):
        _, y_Connectome = self.GCNII_forConcat(adj_t_Connnecton)
        y_Connectome = y_Connectome.log_softmax(dim=-1)
        _, y_Skeleton = self.PointNetSkeleton_Concat(data_Skeleton)
        y_Skeleton = y_Skeleton.log_softmax(dim=-1)
        neuronIDs = data_Skeleton.ID.tolist()

        index_Connectome = []
        for i in range(len(neuronIDs)):
            index_Connectome.append(selected_ID.index(neuronIDs[i]))
        y_selected_Connectome = y_Connectome[index_Connectome]

        # y_selected_Connectome = y_Connectome[selected_ID.index(neuronIDs[0])]
        # y_selected_Connectome = y_selected_Connectome.view(1, -1)
        # for i in range(len(neuronIDs) - 1):
        #     i += 1
        #     index = selected_ID.index(neuronIDs[i])
        #     b = y_Connectome[index].view(1, -1)
        #     y_selected_Connectome = torch.cat((y_selected_Connectome, b), 0)

        y_total = y_selected_Connectome + y_Skeleton

        return y_total
class BrainConnetome_maxPooling_Y(torch.nn.Module):
    def __init__(self, num_nodes, dataset_num_classes, dataset_num_features=1024, represent_features=512,\
                 hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0):
        super().__init__()
        self.represent_features = represent_features
        self.PointNetSkeleton_Concat = PointNetSkeleton_Concat(dataset_num_features, self.represent_features, num_class=dataset_num_classes)
        self.GCNII_forConcat = GCNII_Concat(num_nodes, dataset_num_features, dataset_num_classes, self.represent_features,\
                                            hidden_channels, num_layers, alpha, theta, shared_weights, dropout)

        self.mlp = MLP([2*self.represent_features, (2*self.represent_features - dataset_num_classes) // 3 + dataset_num_classes,\
                        2*(2*self.represent_features - dataset_num_classes) // 3 + dataset_num_classes,\
                        dataset_num_classes], dropout=0.5)


    def forward(self, device, data_Skeleton, adj_t_Connnecton, selected_ID):
        _, y_Connectome = self.GCNII_forConcat(adj_t_Connnecton)
        # y_Connectome = y_Connectome.log_softmax(dim=-1)
        _, y_Skeleton = self.PointNetSkeleton_Concat(data_Skeleton)
        # y_Skeleton = y_Skeleton.log_softmax(dim=-1)
        neuronIDs = data_Skeleton.ID.tolist()

        index_Connectome = []
        for i in range(len(neuronIDs)):
            index_Connectome.append(selected_ID.index(neuronIDs[i]))
        y_selected_Connectome = y_Connectome[index_Connectome]
        # y_selected_Connectome = y_Connectome[selected_ID.index(neuronIDs[0])]
        # y_selected_Connectome = y_selected_Connectome.view(1, -1)
        # for i in range(len(neuronIDs) - 1):
        #     i += 1
        #     index = selected_ID.index(neuronIDs[i])
        #     b = y_Connectome[index].view(1, -1)
        #     y_selected_Connectome = torch.cat((y_selected_Connectome, b), 0)

        y_total = torch.rand(size=(len(neuronIDs), len(y_selected_Connectome[0]))).to(device)
        for i in range(len(neuronIDs)):
            a = y_selected_Connectome[i].view(1, -1)
            b = y_Skeleton[i].view(1, -1)
            temp_total = torch.cat((a, b), 0)
            y_total[i] = torch.max(temp_total, 0)[0]

        y_total = y_total.log_softmax(dim=-1)

        return y_total


class BrainConnetome_cascade_X(torch.nn.Module):
    # pre_train GCNII and concat that
    def __init__(self, dataset_num_classes, dataset_num_features=1024, represent_features=512, num_PointNet2=3):
        super().__init__()
        self.represent_features = represent_features
        self.PointNetSkeleton_Concat = PointNetSkeleton_Concat(dataset_num_features, self.represent_features, num_class=dataset_num_classes, num_PointNet2Layer=num_PointNet2)

        self.mlp = MLP([2*self.represent_features, (2*self.represent_features - dataset_num_classes) // 3 + dataset_num_classes,\
                        2*(2*self.represent_features - dataset_num_classes) // 3 + dataset_num_classes], dropout=0.5)

        self.norm = BatchNorm1d(2*(2*self.represent_features - dataset_num_classes) // 3 + dataset_num_classes)
        self.line = nn.Linear(2*(2*self.represent_features - dataset_num_classes) // 3 + dataset_num_classes, dataset_num_classes)

    def forward(self, hidden_x, data_Skeleton, selected_ID):
        x_Connectome = hidden_x
        x_Skeleton, _ = self.PointNetSkeleton_Concat(data_Skeleton)
        neuronIDs = data_Skeleton.ID.tolist()

        index_Connectome = []
        for i in range(len(neuronIDs)):
            index_Connectome.append(selected_ID.index(neuronIDs[i]))
        x_selected_Connectome = x_Connectome[index_Connectome]
        x_total = torch.cat((x_selected_Connectome, x_Skeleton), 1)
        x_total = self.mlp(x_total)

        y = self.norm(x_total)
        y = F.relu(y)
        y = F.dropout(y, p=0.5, training=self.training)
        y = self.line(y)
        y = y.log_softmax(dim=-1)
        return x_total, y


class BrainConnetome_maxPooling_X(torch.nn.Module):
    # pre_train GCNII and concat that
    def __init__(self, dataset_num_classes, dataset_num_features=1024, represent_features=512):
        super().__init__()
        self.represent_features = represent_features
        self.PointNetSkeleton_Concat = PointNetSkeleton_Concat(dataset_num_features, self.represent_features, num_class=dataset_num_classes)

        self.mlp = MLP([self.represent_features, (self.represent_features - dataset_num_classes) // 3 + dataset_num_classes,\
                        2*(self.represent_features - dataset_num_classes) // 3 + dataset_num_classes], dropout=0.5)

        self.norm = BatchNorm1d(2*(self.represent_features - dataset_num_classes) // 3 + dataset_num_classes)
        self.line = nn.Linear(2*(self.represent_features - dataset_num_classes) // 3 + dataset_num_classes, dataset_num_classes)

    def forward(self, hidden_x, data_Skeleton, selected_ID):
        x_Connectome = hidden_x
        x_Skeleton, _ = self.PointNetSkeleton_Concat(data_Skeleton)
        neuronIDs = data_Skeleton.ID.tolist()

        index_Connectome = []
        for i in range(len(neuronIDs)):
            index_Connectome.append(selected_ID.index(neuronIDs[i]))
        x_selected_Connectome = x_Connectome[index_Connectome]

        x_selected_Connectome = x_selected_Connectome.view(1, -1)
        x_Skeleton = x_Skeleton.view(1, -1)
        x_total = torch.cat((x_selected_Connectome, x_Skeleton), 0)
        maxpool = nn.MaxPool1d(2, stride=1)
        x_total = maxpool(x_total.t()).t()
        x_total = x_total.view(len(index_Connectome), -1)
        x_total = self.mlp(x_total)


        y = self.norm(x_total)
        y = F.relu(y)
        y = F.dropout(y, p=0.5, training=self.training)
        y = self.line(y)
        y = y.log_softmax(dim=-1)
        return x_total, y


class BrainConnetome_avgPooling_X(torch.nn.Module):
    # pre_train GCNII and concat that
    def __init__(self, dataset_num_classes, dataset_num_features=1024, represent_features=512):
        super().__init__()
        self.represent_features = represent_features
        self.PointNetSkeleton_Concat = PointNetSkeleton_Concat(dataset_num_features, self.represent_features, num_class=dataset_num_classes)

        self.mlp = MLP([self.represent_features, (self.represent_features - dataset_num_classes) // 3 + dataset_num_classes,\
                        2*(self.represent_features - dataset_num_classes) // 3 + dataset_num_classes], dropout=0.5)

        self.norm = BatchNorm1d(2*(self.represent_features - dataset_num_classes) // 3 + dataset_num_classes)
        self.line = nn.Linear(2*(self.represent_features - dataset_num_classes) // 3 + dataset_num_classes, dataset_num_classes)

    def forward(self, hidden_x, data_Skeleton, selected_ID):
        x_Connectome = hidden_x
        x_Skeleton, _ = self.PointNetSkeleton_Concat(data_Skeleton)
        neuronIDs = data_Skeleton.ID.tolist()

        index_Connectome = []
        for i in range(len(neuronIDs)):
            index_Connectome.append(selected_ID.index(neuronIDs[i]))
        x_selected_Connectome = x_Connectome[index_Connectome]

        x_selected_Connectome = x_selected_Connectome.view(1, -1)
        x_Skeleton = x_Skeleton.view(1, -1)
        x_total = torch.cat((x_selected_Connectome, x_Skeleton), 0)
        avgpool = nn.AvgPool1d(2, stride=1)
        x_total = avgpool(x_total.t()).t()
        x_total = x_total.view(len(index_Connectome), -1)
        x_total = self.mlp(x_total)


        y = self.norm(x_total)
        y = F.relu(y)
        y = F.dropout(y, p=0.5, training=self.training)
        y = self.line(y)
        y = y.log_softmax(dim=-1)
        return x_total, y

class BrainConnetome_avgPooling_X_for_abalation_PonintNet(torch.nn.Module):
    # pre_train GCNII and concat that
    def __init__(self, dataset_num_classes, dataset_num_features=1024, represent_features=512):
        super().__init__()
        self.represent_features = represent_features
        self.PointNetSkeleton_Concat = PointNetSkeleton_Concat(dataset_num_features, self.represent_features, num_class=dataset_num_classes)

        self.mlp = MLP([self.represent_features, (self.represent_features - dataset_num_classes) // 3 + dataset_num_classes,\
                        2*(self.represent_features - dataset_num_classes) // 3 + dataset_num_classes], dropout=0.5)

        self.norm = BatchNorm1d(2*(self.represent_features - dataset_num_classes) // 3 + dataset_num_classes)
        self.line = nn.Linear(2*(self.represent_features - dataset_num_classes) // 3 + dataset_num_classes, dataset_num_classes)

    def forward(self, data_Skeleton, selected_ID):
        x_Skeleton, _ = self.PointNetSkeleton_Concat(data_Skeleton)
        neuronIDs = data_Skeleton.ID.tolist()

        index_Connectome = []
        for i in range(len(neuronIDs)):
            index_Connectome.append(selected_ID.index(neuronIDs[i]))

        x_Skeleton = x_Skeleton.view(1, -1)
        avgpool = nn.AvgPool1d(2, stride=1)
        x_total = x_total.view(len(index_Connectome), -1)
        x_total = self.mlp(x_total)


        y = self.norm(x_total)
        y = F.relu(y)
        y = F.dropout(y, p=0.5, training=self.training)
        y = self.line(y)
        y = y.log_softmax(dim=-1)
        return x_total, y










