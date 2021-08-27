class Config:
    # data
    encoding = 'embedding'
    max_length = 900

    # model
    src_vocab = 24  # 21 + 3
    # emb_dim = 21
    d_model = 128  # hidden layer
    trg_vocab = 245
    heads = 8
    n_layers = 6
    load_weights = None
    dropout = 0.5

    # optimizer
    lr = 1e-4

    # training
    batch_size = 8
    epoch = 10


opt = Config()
