import numpy as np
from torch.utils.data import Dataset
train_path = 'C:\\Users\\zhangpengyan\\Downloads\\seq2fold\\train\\astral_train.fa'
test_path = 'C:\\Users\\zhangpengyan\\Downloads\\seq2fold\\test\\astral_test.fa'


def fasta2dict(inf, label_mode=True):
    name = None
    data = {}
    for line in inf:
        line = line.strip()
        if line.startswith('>'):
            name = line.split()[0][1:]
            data[name] = {}
            data[name]['seq'] = ''
            if label_mode:
                label_split = line.split()[1].split('.')
                label = label_split[0]+'.'+label_split[1]
                data[name]['label'] = label
        else:
            data[name]['seq'] += line
    return data


class SeqDataset(Dataset):
    def __init__(self, train=True):
        self.train = train
        self.data_path = train_path if self.train else test_path
        self.label_map = np.loadtxt('label.txt', dtype=str)
        with open(self.data_path, 'r') as f:
            self.data = fasta2dict(f, label_mode=train)
        self.length = len(self.data)
        print(f"load {'train' if self.train else 'test'} dataset size: {self.length}")

    def __getitem__(self, item):
        name = list(self.data.keys())[item]
        seq = self.data[name]['seq']
        if self.train:
            label = self.data[name]['label']
            return name, seq, int(np.where(self.label_map == label)[0])
        else:
            return name, seq

    def __len__(self):
        return self.length


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data = SeqDataset(train=True)
    dataloader = DataLoader(dataset=data, batch_size=2, shuffle=False, num_workers=1)
    for name, seq, label in dataloader:
        print(name, seq, label)
