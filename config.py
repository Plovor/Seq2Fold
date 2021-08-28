class Config:
    # data
    encoding = 'embedding'  # embedding + position
    max_length = 800

    # model
    src_vocab = 24  # embedding layer, 21 + 3
    trg_vocab = 245  # class num
    emb_dim = 128  # embedding layer
    d_model = 128  # hidden layer
    n_layers = 6  # hidden layer
    heads = 8  # multi-head attention
    load_weights = None
    dropout = 0.5

    # optimizer
    lr = 1e-4

    # training
    batch_size = 8
    epoch = 10


opt = Config()
