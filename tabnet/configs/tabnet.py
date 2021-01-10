import os


class config:
    _workspace_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    _workspace_path = os.path.abspath(_workspace_path)

    name = '2020_11_22_0033'
    description = 'Adam, nstep1'
    seed = [42]
    device = [0]
    data_dir = os.path.join(_workspace_path, 'dataset', 'lish-moa')
    log_dir = os.path.join(_workspace_path, 'logs', 'lish-moa', name)
    bkup_dir = os.path.join(_workspace_path, 'repo', 'lish-moa')
    sub_dir = os.path.join(_workspace_path, 'notebook', 'lish-moa', 'csv')

    num_scored = 206
    num_nonscored = 402

    # preprocess
    normalize = True
    remove_vehicle = True
    use_autoencoder = False

    # from https://www.kaggle.com/vbmokin/moa-pytorch-rankgauss-pca-nn-upgrade-3d-visual
    # of version 27
    feature_status = False
    rank_gauss = True
    pca = True
    cell_squared = False
    scaling = False
    vt_selection = True
    lsvc_selection = False

    n_comp_GENES = 463
    n_comp_CELLS = 60
    VarianceThreshold_for_FS = 0.84
    QT_n_quantile_min = 10
    QT_n_quantile_max = 200

    # Model
    n_d = 128
    n_a = 32
    n_independent = 1   # feature extractor layer-wise cell
    n_shared = 1        # feature extractor shared cell
    cat_emb_dim = 4     # 논문에 따로 내용이 없는 부분; 0 이면 사용 안함
    n_steps = 1
    gamma = 1.0
    lambda_sparse = 0.

    # Training
    n_folds = 5
    n_pretrain_epochs = 100
    n_epochs = 200
    batch_size = 1024
    virtual_batch_size = 128
    momentum = 0.02

    lr = 2e-2
    weight_decay = 1e-5
    scheduler_step_size = 50
    scheduler_gamma = 0.9
    num_workers = 8
