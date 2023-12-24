import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from model.model import GCNII_Concat, BrainConnetome_cascade_X
from datasets.HemiConnectome import HemiConnectome, HemiSkeleton
from datasets.H01Connectome import H01Connectome
from datasets.H01Skeleton import H01Skeleton
import argparse
from torch_geometric.loader import DataLoader
import random

parser = argparse.ArgumentParser(
    description='train',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#### about train ####
parser.add_argument('--deviceID', type=int, default=0)
parser.add_argument('--train_patient', type=int, default=15)
parser.add_argument('--train_gcn_patient', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--train_lr', type=int, default=0.01)
parser.add_argument('--GCNII_Graph_weight_decay', type=float, default=0.01)
parser.add_argument('--GCNII_linear_weight_decay', type=float, default=5e-4)

args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_gcn(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    hidden_x, out = model(data.adj_t)
    out = out.log_softmax(dim=-1)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return hidden_x, float(loss)

@torch.no_grad()
def test_gcn(model, data):
    model.eval()
    _, pred = model(data.adj_t)
    pred = pred.log_softmax(dim=-1)
    pred = pred.argmax(dim=-1)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

def train(model, hidden_x, train_loader, selected_ID, device, optimizer):
    model.train()
    loss_sum = 0
    corrects = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        _, ypre = model(hidden_x, data, selected_ID)
        loss = F.nll_loss(ypre, data.y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        with torch.no_grad():
            pred = ypre.max(1)[1]
        correct = pred.eq(data.y).sum().item()
        corrects += correct
    train_acc = corrects / len(train_loader.dataset)
    return train_acc

def test(model, hidden_x, test_loader, selected_ID, device):
    model.eval()
    corrects = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            _, pred = model(hidden_x, data, selected_ID)
            pred = pred.max(1)[1]
        correct = pred.eq(data.y).sum().item()
        corrects += correct
    test_acc = corrects / len(test_loader.dataset)
    # wandb.log({'test_acc': test_acc})
    return test_acc

def main(seed, data):
    setup_seed(seed)
    device = torch.device(f'cuda:{args.deviceID}' if torch.cuda.is_available() else 'cpu')

    if data == 'HemiBrain':
        path_Skeletome = 'data/HemiBrain/Skeleton'
        path_Connectome = 'data/HemiBrain/Connectome'
        pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)
        data_dataset = HemiSkeleton(path_Skeletome, 'data', transform=transform, pre_transform=pre_transform)
        train_dataset = HemiSkeleton(path_Skeletome, 'train', transform=transform, pre_transform=pre_transform)
        test_dataset = HemiSkeleton(path_Skeletome, 'test', transform=transform, pre_transform=pre_transform)
        data_loader = DataLoader(data_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)
        transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
        dataset_Connectome = HemiConnectome(path_Connectome, transform=transform)

    if data == 'H01':
        path_Skeletome = 'data/H01/Skeleton'
        path_Connectome = 'data/H01/Connectome'
        pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)
        data_dataset = H01Skeleton(path_Skeletome, 'data', transform=transform, pre_transform=pre_transform)
        train_dataset = H01Skeleton(path_Skeletome, 'train', transform=transform, pre_transform=pre_transform)
        test_dataset = H01Skeleton(path_Skeletome, 'test', transform=transform, pre_transform=pre_transform)
        data_loader = DataLoader(data_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)
        transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
        dataset_Connectome = H01Connectome(path_Connectome, transform=transform)

    data_Connectome = dataset_Connectome[0]
    data_Connectome = data_Connectome.to(device)
    selected_ID = data_Connectome.selected_ID
    data_Connectome.adj_t = gcn_norm(data_Connectome.adj_t)
    num_nodes, dataset_num_classes = len(dataset_Connectome.data.y), data_dataset.num_classes
    model_gnn = GCNII_Concat(num_nodes, dataset_num_features=1024, num_calss=dataset_num_classes + 1, represent_features=512,\
                             hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0).to(device)
    optimizer_gcn = torch.optim.Adam([
        dict(params=model_gnn.convs.parameters(), weight_decay=args.GCNII_Graph_weight_decay),
        dict(params=model_gnn.lins.parameters(), weight_decay=args.GCNII_linear_weight_decay),
        dict(params=model_gnn.mlp.parameters(), weight_decay=args.GCNII_linear_weight_decay)
    ], lr=args.train_lr)

    model = BrainConnetome_cascade_X(dataset_num_classes, dataset_num_features=1024, represent_features=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_test_acc = 0
    epoch = 0
    bad = 0
    patient = args.train_gcn_patient
    print('GCNII is pre_trainning...')
    while True:
        if bad > patient:
            break
        epoch += 1
        bad += 1
        hidden_x, loss = train_gcn(model_gnn, data_Connectome, optimizer_gcn)
        _, _, tmp_test_acc = test_gcn(model_gnn, data_Connectome)
        if tmp_test_acc > max_test_acc:
            bad = 0
            max_test_acc = tmp_test_acc
            print(f'Epoch: {epoch:03d}, loss: {loss:.6f}, Test_acc: {tmp_test_acc:.4f}')
            continue
        if bad % (args.train_gcn_patient//5) == 0:
            print(f'Epoch: {epoch:03d}')

    max_test_acc = 0
    epoch = 0
    bad = 0
    hidden_x = torch.from_numpy(hidden_x.cpu().detach().numpy()).to(device)
    print('main train is working...')
    while True:
        if epoch == args.train_patient:
            break
        bad += 1
        epoch += 1
        train_acc_end = train(model, hidden_x, train_loader, selected_ID, device, optimizer)
        test_acc = test(model, hidden_x, test_loader, selected_ID, device)
        if test_acc > max_test_acc:
            bad = 0
            max_test_acc = test_acc
            print(f'Epoch: {epoch:03d}, loss: {loss:.6f}, Train: {train_acc_end:.4f}, Test: {test_acc:.4f}')
            continue
        if bad % (args.train_patient//5) == 0:
            print(f'Epoch: {epoch:03d}')
    return max_test_acc


if __name__ == '__main__':
    for (data, seed) in [('HemiBrain', 666), ('H01',333)]:
            print(f'##################   {data}: {seed} is working...     ##################')
            max_test_acc = main(seed, data)
            print(f'max_test_acc: {max_test_acc:.4f}')











