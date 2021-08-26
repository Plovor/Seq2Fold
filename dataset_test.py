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


def seq_encoding(seq: str, mode: str):
    n = len(seq)
    coding = None
    if mode is None:
        mode = 'blosum'
    if mode == 'blosum':
        symbol = np.loadtxt('BLOSUM62.txt', max_rows=1, dtype=str)
        matrix = np.loadtxt('BLOSUM62.txt', skiprows=1, dtype=float)
        # print(symbol, matrix.shape, matrix[0, :])
        coding = np.zeros([n, symbol.shape[0]])
        for i in range(n):
            a = np.where(symbol == seq[i])[0]
            if a.size == 0:
                a = np.where(symbol == chr(ord(seq[i])+32))[0]
            coding[i, :] = matrix[a, :]
    return coding


class SeqDataset(Dataset):
    def __init__(self, train=True, encoding='blosum'):
        self.train = train
        self.encoding = encoding
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
            return name, seq_encoding(seq, self.encoding), int(np.where(self.label_map == label)[0])
        else:
            return name, seq_encoding(seq, self.encoding)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data = SeqDataset(train=True, encoding='blosum')
    dataloader = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=1)
    label_list = []
    l = []
    for name, seq, label in dataloader:
        l.append(seq.shape[1])
    print(max(l), min(l))
        #print(name, seq.shape, label)
    #     label_list.append(label[0])
    # label_array = np.array(label_list)
    # label_unique = np.unique(label_array)
    # print(label_unique.shape)
    # np.savetxt('label.txt', label_unique, fmt='%s')
