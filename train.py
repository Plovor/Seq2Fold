import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
import numpy as np
from dataset import SeqDataset
from transformer.Models import get_model
from torch.utils.data import DataLoader, random_split
from config import opt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, dataset, epoch=opt.epoch, batch_size=opt.batch_size):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    pbar = tqdm(train_loader)
    for e in range(epoch):
        for name, seq, label in pbar:
            y_predict = net()
            y_target = label.to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.set_description('train epoch {}'.format(epoch))

    return


if __name__ == '__main__':
    train(get_model(opt, 1, 1), SeqDataset(train=True))

