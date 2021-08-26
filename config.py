class Config:
    # model
    vocab_size = 21
    d_model = 21
    heads = 1
    n_layers = 1
    load_weights = None
    device = 0
    dropout = 0.5

    # optimizer
    lr = 1e-4

    # training
    batch_size = 8
    epoch = 10


opt = Config()
