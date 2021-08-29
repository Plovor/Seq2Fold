import numpy as np
from torch.utils.data import Dataset
train_path = 'data/astral_train.fa'
test_path = 'data/astral_test.fa'


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


def load_blosum(path):
    blosum = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            aa = line.split()[0]
            feature = line.split()[1:]
            feature = np.array(feature, dtype=float)
            blosum[aa] = feature
    return blosum


class SeqDataset(Dataset):
    def __init__(self, opt, train=True):
        self.max_length = opt.max_length
        self.train = train
        self.alphabet = {aa: i for i, aa in enumerate(opt.alphabet)}  # XOUBZ -> *
        self.blosum = load_blosum('BLOSUM62.txt')
        self.data_path = train_path if self.train else test_path
        self.label_map = np.loadtxt('label.txt', dtype=str)
        self.class_num = self.label_map.shape[0]
        with open(self.data_path, 'r') as f:
            self.data = fasta2dict(f, label_mode=train)
        self.length = len(self.data)
        print(f"load {'train' if self.train else 'test'} dataset size: {self.length}")

    def _encode_seq(self, input_seq: str):
        coding = np.zeros([self.max_length])  # [PAD] = 0
        coding[0] = 1  # [CLS] = 1
        coding[-1] = 2  # [SEP] = 2
        input_seq = input_seq[:self.max_length - 2]
        for i, aa in enumerate(input_seq):
            if aa in self.alphabet.keys():
                coding[i + 1] = self.alphabet[aa] + 3
            elif chr(ord(aa) - 32) in self.alphabet.keys():
                coding[i + 1] = self.alphabet[chr(ord(aa) - 32)] + 3
            else:
                coding[i + 1] = self.alphabet['*'] + 3

        return coding

    # def _encode_label(self, input_label: int):
    #     output_label = [0 for _ in range(self.class_num)]
    #     output_label[input_label] = 1
    #     return np.array(output_label, dtype=int)

    def __getitem__(self, item):
        name = list(self.data.keys())[item]
        seq = self.data[name]['seq']
        if self.train:
            label = self.data[name]['label']
            label = int(np.where(self.label_map == label)[0])
            # return name, self._encode_seq(seq), self._encode_label(label)
            return name, self._encode_seq(seq), label
        else:
            return name, self._encode_seq(seq)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from config import opt
    data = SeqDataset(opt, train=False)
    dataloader = DataLoader(dataset=data, batch_size=2, shuffle=False, num_workers=1)
    for n, s in dataloader:
        print(s.shape)
        break
