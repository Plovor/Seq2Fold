import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable

from config import opt

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_emb):
        super().__init__()
        self.d_model = d_emb
        self.embed = nn.Embedding(vocab_size, d_emb)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_emb, max_seq_len=opt.max_length, dropout=opt.dropout):
        super().__init__()
        self.d_model = d_emb
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_emb)
        for pos in range(max_seq_len):
            for i in range(0, d_emb, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i)/d_emb)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1))/d_emb)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


class BlosumEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blosum = self._load_blosum(opt.blosum_path, opt.alphabet)

    def forward(self, src, x):
        coding = np.zeros([src.shape[0], src.shape[1], self.blosum[3].shape[0]])
        for b in range(src.shape[0]):
            for i, aa in enumerate(src[b, :]):
                if aa in self.blosum.keys():
                    coding[b, i, :] = self.blosum[aa]
                else:
                    coding[b, i, :] = self.blosum[opt.src_vocab - 1]
        be = Variable(torch.from_numpy(coding).to(torch.float32), requires_grad=False)
        if x.is_cuda:
            be = be.cuda()
        x = torch.cat((x, be), dim=2)
        return x

    @staticmethod
    def _load_blosum(path, alphabet):
        blosum = {}
        alphabet_dic = {aa: i+3 for i, aa in enumerate(alphabet)}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                aa = line.split()[0]
                feature = line.split()[1:]
                feature = np.array(feature, dtype=float)
                if aa in alphabet_dic.keys():
                    blosum[alphabet_dic[aa]] = feature
        return blosum
