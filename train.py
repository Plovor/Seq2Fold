import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
import numpy as np
from dataset import SeqDataset
from transformer.Models import get_model
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from config import opt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, dataset, epoch=opt.epoch, batch_size=opt.batch_size):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    for e in range(epoch):
        pbar = tqdm(train_loader)
        print('train epoch {}'.format(e))
        for name, seq, label in pbar:
            inputs = seq.long().to(device)
            y_predict = net(inputs)
            y_target = label.long().to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_predict, y_target)
            loss.backward()
            optimizer.step()
            # descri = ('%10s' * 2 + '%10.3g' * 5) % ('%g/%g' % (epoch, opt.epoch - 1), mem, *mloss)
            pbar.set_description('loss: %10.3g' % loss)

    return


if __name__ == '__main__':
    model = get_model(opt, 24, 245)
    model = model.to(device)
    data = SeqDataset(opt, train=True)
    train(model, data)

