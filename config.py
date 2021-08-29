class Config:

    # model
    src_vocab = 24  # embedding layer, 21 + 3
    class_num = 245  # class num
    d_model = 128  # hidden layer
    n_layers = 6  # hidden layer
    heads = 8  # multi-head attention
    load_weights = None
    dropout = 0.1

    # encoding: blosum
    blosum = False
    blosum_path = 'BLOSUM62.txt'

    # encoding: acc
    acc = False

    # data
    max_length = 800
    alphabet = 'ARNDCQEGHILKMFPSTWYV*'

    # optimizer
    lr = 1e-4

    # training
    batch_size = 8
    epoch = 100


opt = Config()
