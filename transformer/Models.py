import torch
import torch.nn as nn
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.Embed import Embedder, PositionalEncoder, BlosumEncoder
from transformer.Sublayers import Norm
import copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, blosum, N, heads, dropout):
        super().__init__()
        self.N = N
        d_emb = d_model - 24 if blosum else d_model
        self.embed = Embedder(vocab_size, d_emb)
        self.pe = PositionalEncoder(d_emb)
        self.be = BlosumEncoder() if blosum else None
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        if self.be is not None:
            x = self.be(src, x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


class SeqEncoder(nn.Module):
    def __init__(self, src_vocab, class_num, d_model, blosum, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, blosum, N, heads, dropout)
        self.out = nn.Linear(d_model, class_num)

    def forward(self, src, src_mask=None):
        e_outputs = self.encoder(src, src_mask)
        e_outputs = e_outputs[:, 0, :]
        output = self.out(e_outputs)
        return output


def get_model(opt):

    assert opt.dropout < 1

    model = SeqEncoder(opt.src_vocab, opt.class_num, opt.d_model,
                       opt.blosum, opt.n_layers, opt.heads, opt.dropout)

    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model
