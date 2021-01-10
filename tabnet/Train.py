import os
from distutils.dir_util import copy_tree
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from dataset.moaDataset import moaDataset, cate2num
from models.moaModel import moaModel
from scripts.utils import seed_everything
from configs.mlp import config
from Learner import Learner
from Infer import Infer


if __name__ == '__main__':
    # Backup Code
    dst_dir = os.path.join(config.log_dir, 'code')
    if not os.path.isdir(dst_dir):
        copy_tree(config.bkup_dir, dst_dir)

    # Load Data
    train_features = pd.read_csv(f'{config.data_dir}/train_features_with_{config.n_folds}folds.csv')
    train_targets_scored = pd.read_csv(f'{config.data_dir}/train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv(f'{config.data_dir}/train_targets_nonscored.csv')
    test_features = pd.read_csv(f'{config.data_dir}/test_features.csv')
    submission = pd.read_csv(f'{config.data_dir}/sample_submission.csv')

    # Preprocess Data
    train_features = train_features[train_features.cp_type != 'ctl_vehicle']
    test_features = test_features[test_features.cp_type != 'ctl_vehicle']
    train_features = train_features.merge(train_targets_scored, on='sig_id')
    train_features = train_features.merge(train_targets_nonscored, on='sig_id')
    train_features = cate2num(train_features)
    test_features = cate2num(test_features)

    if config.feature:
        train_features = pd.read_csv(f'{config.data_dir}/features/train_{config.n_folds}folds_{config.feature}.csv')
        test_features = pd.read_csv(f'{config.data_dir}/features/test_{config.n_folds}folds_{config.feature}.csv')

    # Column Infos
    score_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
    nonscore_cols = [c for c in train_targets_nonscored.columns if c not in ['sig_id']]
    cate_cols = ['cp_time', 'cp_dose']
    g_cols = [c for c in test_features.columns if 'g-' in c]
    c_cols = [c for c in test_features.columns if 'c-' in c]
    num_g = len(g_cols)
    num_c = len(c_cols)
    num_cate = len(cate_cols)

    # Training
    saved_models = {'cv': [], 'path': []}
    for seed in config.seed:
        seed_everything(seed)
        hists = {'train': [], 'valid': []}
        for fold in range(config.n_folds):
            print(f'[[ SEED {seed} Fold {fold} ]]')
            fold_train = train_features[train_features.fold != fold].reset_index(drop=True)
            fold_valid = train_features[train_features.fold == fold].reset_index(drop=True)

            train_dataset = moaDataset(fold_train.sig_id.values,
                                       fold_train[g_cols].values,
                                       fold_train[c_cols].values,
                                       fold_train[cate_cols].values,
                                       fold_train[score_cols].values,
                                       fold_train[nonscore_cols].values)
            valid_dataset = moaDataset(fold_valid.sig_id.values,
                                       fold_valid[g_cols].values,
                                       fold_valid[c_cols].values,
                                       fold_valid[cate_cols].values,
                                       fold_valid[score_cols].values,
                                       fold_valid[nonscore_cols].values)

            train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                      shuffle=True, num_workers=8, drop_last=True)
            valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)

            model = moaModel(config, num_g, num_c, num_cate,
                             classes=config.num_scored,
                             non_classes=config.num_nonscored)
            train_module = Learner(model, train_loader, valid_loader, fold, config, seed)
            history = train_module.fit(config.n_epochs)

            for key in hists.keys():
                hists[key].append(history[key])
            saved_models['cv'].append(min(history['valid']))
            saved_models['path'].append(os.path.join(config.log_dir, f"{config.name}_seed{seed}_fold{fold}.pth"))

        # Inference
        cv_seed = np.mean([min(hist_valid) for hist_valid in hists['valid']])
        print(f'SEED {seed}: CV {cv_seed:.6f}')

    cv_score = np.mean(saved_models['cv'])
    dataset = moaDataset(test_features.sig_id.values,
                         test_features[g_cols].values,
                         test_features[c_cols].values,
                         test_features[cate_cols].values,
                         test_features.sig_id.values,
                         test=True)

    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
    pred_ls = []
    for path in saved_models['path']:
        model = moaModel(config, num_g, num_c, num_cate,
                         classes=config.num_scored,
                         non_classes=config.num_nonscored)
        infer_module = Infer(model, test_loader, config)
        infer_module.load(path)
        pred = infer_module.inference()
        pred_ls.append(pred)
    results = np.mean(pred_ls, axis=0)
    test_results = test_features.copy()
    test_results[score_cols] = results
    sub = submission.drop(columns=score_cols).merge(test_results[['sig_id'] + score_cols],
                                                    on='sig_id', how='left').fillna(0)
    sub.to_csv(os.path.join(config.sub_dir, f'{config.name}_{cv_score:.6f}.csv'), index=False)
    print(f'[cv:{cv_score:.6f}] {config.name} Finished')

    cv_seeds = [saved_models['cv'][i:i + config.n_folds] for i in range(0, len(saved_models['cv']), config.n_folds)]
    for idx in range(len(cv_seeds)):
        print(f'SEED {config.seed[idx]} CV score: {np.mean(cv_seeds[idx]):.6f}')
