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


def get_acc(predict, target):
    predict = predict.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    pre_max = np.argmax(predict, axis=1)
    acc = np.sum(pre_max == target, axis=0) / target.shape[0]
    return acc


def train(net, dataset, epoch=opt.epoch, batch_size=opt.batch_size):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    for e in range(epoch):
        pbar = tqdm(train_loader)
        running_loss = 0
        running_batch = 0
        running_acc = 0
        print('train epoch {}'.format(e))
        for i, (_, seq, label) in enumerate(pbar):
            inputs = seq.long().to(device)
            y_predict = net(inputs)
            y_target = label.long().to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_predict, y_target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.shape[0] / len(train_loader)
            running_acc += get_acc(y_predict, y_target) * inputs.shape[0] / len(train_loader)
            running_batch += inputs.shape[0]
            descri_loss = running_loss * len(train_loader) / running_batch
            descri_acc = running_acc * len(train_loader) / running_batch
            pbar.set_description('loss: %10.4g, acc: %10.4g' % (descri_loss, descri_acc))

        pbar = tqdm(val_loader)
        running_loss = 0
        running_batch = 0
        running_acc = 0
        print('val')
        for i, (_, seq, label) in enumerate(pbar):
            inputs = seq.long().to(device)
            y_predict = net(inputs)
            y_target = label.long().to(device)
            loss = F.cross_entropy(y_predict, y_target)
            running_loss += loss.item() * inputs.shape[0] / len(val_loader)
            running_acc += get_acc(y_predict, y_target) * inputs.shape[0] / len(val_loader)
            running_batch += inputs.shape[0]
            descri_loss = running_loss * len(val_loader) / running_batch
            descri_acc = running_acc * len(val_loader) / running_batch
            pbar.set_description('loss: %10.4g, acc: %10.4g' % (descri_loss, descri_acc))
    return


if __name__ == '__main__':
    model = get_model(opt, 24, 245)
    model = model.to(device)
    data = SeqDataset(opt, train=True)
    train(model, data)

