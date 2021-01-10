import torch


class config:
    name = 'ckpt_mlp'
    seed = [42, 43, 45]
    device = torch.device('cuda:0')
    data_dir = '/workspace/dataset/lish-moa'
    log_dir = f'/workspace/logs/lish-moa/{name}'
    bkup_dir = '/workspace/repo/lish-moa/'
    sub_dir = '/workspace/notebook/lish-moa/csv'

    # feature = '20201108_0640'
    feature = None

    num_scored = 206
    num_nonscored = 402

    # Model
    dropout = 0.5
    hidden_size = 512

    # Training
    n_folds = 5
    n_epochs = 20
    batch_size = 32
    fp16 = False
    smoothing = 0.001
    lr = 1e-2
    weight_decay = 1e-6

    # Test
    tta = True

    desc = "3 seed / 5 fold / g-c(comb)"
