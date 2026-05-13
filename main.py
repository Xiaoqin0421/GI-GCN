import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import time
import random

from model import GI_GCN


def get_argparse():
    parser = argparse.ArgumentParser(description='Global Interacted Graph Convolutional Networks')

    parser.add_argument('--dataset', type=str, help='TUDataset name (e.g., MUTAG, PROTEINS)')
    parser.add_argument('--cuda', type=int, help='GPU id')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--hidden', type=int, help='Hidden dim')
    parser.add_argument('--out-dim', type=int, help='Output embedding dim before pooling')
    parser.add_argument('--num-classes', type=int, help='Number of classes')
    parser.add_argument('--power-iter', type=int, help='DS-GCN power iteration steps')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='L2')
    parser.add_argument('--dropout', type=float, help='Dropout')
    parser.add_argument('--epochs', type=int, help='Training epochs')

    parser.set_defaults(
        dataset='IMDB-BINARY',
        cuda=0,
        batch_size=64,
        hidden=32,
        out_dim=32,
        num_classes=2,
        power_iter=3,
        lr=0.001,
        weight_decay=0.01,
        dropout=0.5,
        epochs=100,
    )
    return parser.parse_args()

class GI_GCN_Net(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, num_classes, power_iter, dropout):
        super().__init__()

        self.conv = GI_GCN(
            in_channels,
            hidden,
            out_channels,
            max_iter=power_iter,
            eps=1e-6,
            dropout=dropout
        )
        self.classifier1 = torch.nn.Linear(out_channels, 256)
        self.classifier2 = torch.nn.Linear(256, num_classes)


    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index, batch)

        # Mean pooling
        x = global_mean_pool(x, batch)

        x = self.classifier1(x)
        x = self.classifier2(x)
        return x



def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_time = 0
    for data in loader:
        begin_time = time.time()
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        elapsed = time.time() - begin_time
        total_time += elapsed
    return total_loss / len(loader), total_time

def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

def set_seed(seed):


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = get_argparse()
    print(args)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    dataset = TUDataset(root='./data', name=args.dataset)

    num_classes = dataset.num_classes
    in_dim = dataset.num_features
    max_deg = 20
    new_dataset = []
    for data in dataset:
        if data.x is None:
            in_dim = max_deg + 1
            degs = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
            degs[degs > max_deg] = max_deg
            x = torch.zeros((data.num_nodes, max_deg + 1))
            x[torch.arange(data.num_nodes), degs] = 1
            data.x = x
        new_dataset.append(data)
    dataset = new_dataset

    set_seed(111)

    labels = [data.y.item() for data in dataset]
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    train_idx, test_idx = next(skf.split(dataset, labels))
    train_dataset = [dataset[i] for i in train_idx]
    test_dataset = [dataset[i] for i in test_idx]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = GI_GCN_Net(
        in_channels=in_dim,
        hidden=args.hidden,
        out_channels=args.out_dim,
        num_classes=num_classes,
        power_iter=args.power_iter,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(args.epochs):
        loss, epoch_time = train(model, train_loader, optimizer, device)

        scheduler.step()
        if epoch % 10 == 0:
            model.eval()
            acc = test(model, train_loader, device)
            print('Epoch: ', epoch, ' Avg loss: ', loss, '; train acc: ', acc,
                  '; epoch time: ', epoch_time)
    test_acc = test(model, test_loader, device)

    print("test_acc: ", test_acc)



if __name__ == '__main__':
    main()
