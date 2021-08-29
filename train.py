import torch
import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
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

    min_loss = np.inf
    running_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    encoding = 'blosum' if opt.blosum else 'embedding'
    model_path = 'model/' + str(batch_size) + encoding + running_time
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    log_dic = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for e in range(epoch):

        # TRAIN
        pbar = tqdm(train_loader)
        running_loss = 0
        running_batch = 0
        running_acc = 0
        for i, (_, seq, label) in enumerate(pbar):
            inputs = seq.long().to(device)
            y_predict = net(inputs)
            y_target = label.long().to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_predict, y_target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.shape[0] / train_size
            running_acc += get_acc(y_predict, y_target) * inputs.shape[0] / train_size
            running_batch += inputs.shape[0]
            descri_loss = running_loss * train_size / running_batch
            descri_acc = running_acc * train_size / running_batch
            pbar.set_description('train epoch: %d, loss: %8.4g, acc: %8.4g' % (e, descri_loss, descri_acc))
        log_dic['train_loss'].append(running_loss)
        log_dic['train_acc'].append(running_acc)

        # VALIDATE
        pbar = tqdm(val_loader)
        running_loss = 0
        running_batch = 0
        running_acc = 0
        for i, (_, seq, label) in enumerate(pbar):
            inputs = seq.long().to(device)
            y_predict = net(inputs)
            y_target = label.long().to(device)
            loss = F.cross_entropy(y_predict, y_target)
            running_loss += loss.item() * inputs.shape[0] / val_size
            running_acc += get_acc(y_predict, y_target) * inputs.shape[0] / val_size
            running_batch += inputs.shape[0]
            descri_loss = running_loss * val_size / running_batch
            descri_acc = running_acc * val_size / running_batch
            pbar.set_description('val epoch: %d, loss: %8.4g, acc: %8.4g' % (e, descri_loss, descri_acc))
        log_dic['val_loss'].append(running_loss)
        log_dic['val_acc'].append(running_acc)

        if min_loss > running_loss:
            min_loss = running_loss
            torch.save(net.state_dict(), os.path.join(model_path, 'best_model.pth'))

    # SAVE LOG
    log_dt = pd.DataFrame(log_dic)
    log_dt.to_csv(os.path.join(model_path, 'log.csv'), index=False)

    return


if __name__ == '__main__':
    model = get_model(opt)
    model = model.to(device)
    data = SeqDataset(opt, train=True)
    train(model, data)

