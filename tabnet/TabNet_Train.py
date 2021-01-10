import os
from shutil import copytree, ignore_patterns
import numpy as np
import pandas as pd
import pickle

from dataset.moaDataset import DatasetWithLabel, DatasetWithoutLabel, collate_fn_train, collate_fn_test
from dataset.preprocess import cate2num, preprocess_fn
from models.module import SmoothBCEwLogits
from scripts.utils import seed_everything
from scripts.metric import LogitsLogLoss
from scripts.utils import init_logger
from configs.tabnet import config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, OneCycleLR
from sklearn.metrics import roc_auc_score
from pytorch_tabnet.tab_network import TabNet

import warnings
warnings.filterwarnings('ignore')


def get_tabnet(
        input_dim, output_dim, n_d, n_a, n_steps, gamma,
        remove_vehicle, cat_emb_dim, n_independent, n_shared,
        virtual_batch_size, momentum, epsilon=1e-8, mask_type="sparsemax",
        ):
    if remove_vehicle:
        cat_idxs = [0, 1]
        cat_dims = [3, 2]   # Fixed
    else:
        cat_idxs = [0, 1, 2]
        cat_dims = [2, 3, 2]

    if cat_emb_dim == 0:
        cat_idxs = []
        cat_dims = []

    network = TabNet(
        input_dim,
        output_dim,
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dim,
        n_independent=n_independent,
        n_shared=n_shared,
        epsilon=epsilon,
        virtual_batch_size=virtual_batch_size,
        momentum=momentum,
        mask_type=mask_type,
    )
    return network


def train_epoch(model, train_loader, optimizer, scheduler, lambda_, device):
    model.train()
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = SmoothBCEwLogits(
        smoothing=0.001
    )
    for feature, target in train_loader:
        if device:
            feature = feature.to(device)
            target = target.to(device)

        pred, sparsity_loss = model(feature)
        loss_ = loss_fn(pred, target) + lambda_ * sparsity_loss

        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        if scheduler.__class__.__name__ != 'ReduceLROnPlateau':
            scheduler.step()

        yield loss_.item()


def auc_multi(y_true, y_pred):
    M = y_true.shape[1]
    results = np.zeros(M)
    for i in range(M):
        try:
            results[i] = roc_auc_score(y_true[:, i], y_pred[:, i])
        except:
            pass
    return results.mean()


def validation(model, valid_loader, device):
    model.eval()
    logits = []
    preds = []
    targets = []
    with torch.no_grad():
        for feature, target in valid_loader:
            if device:
                feature = feature.to(device)
            logit, _ = model(feature)
            pred = torch.sigmoid(logit)
            logits.append(logit.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            targets.append(target)
    logits = np.concatenate(logits, axis=0)
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    log_logit = LogitsLogLoss()(targets, logits)
    mean_auc = auc_multi(targets, preds)
    return log_logit, mean_auc


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for feature in loader:
            if device:
                feature = feature.to(device)
            logit, _ = model(feature)
            pred = torch.sigmoid(logit)
            preds.append(pred.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return preds


# ref: https://www.kaggle.com/hiramcho/moa-tabnet-with-pca-rank-gauss?select=gauss_rank_scaler.py
if __name__ == '__main__':
    # Backup Code
    dst_dir = os.path.join(config.log_dir, 'code')
    if not os.path.isdir(dst_dir):
        copytree(config.bkup_dir, dst_dir, ignore=ignore_patterns('.*'))

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.device))
    main_logger = init_logger(config.log_dir, f'train_main.log')
    main_logger.info('\n'.join([f"{k} = {v}" for k, v in config.__dict__.items()]))

    # Load Data
    train_features = pd.read_csv(f'{config.data_dir}/train_features.csv')
    train_targets_scored = pd.read_csv(f'{config.data_dir}/train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv(f'{config.data_dir}/train_targets_nonscored.csv')
    test_features = pd.read_csv(f'{config.data_dir}/test_features.csv')
    submission = pd.read_csv(f'{config.data_dir}/sample_submission.csv')

    # Feature Engineering
    train_features, test_features = preprocess_fn(train_features, test_features, train_targets_scored, config)

    # Preprocess Data
    if config.remove_vehicle:
        train_features = train_features[train_features.cp_type != 'ctl_vehicle']
        test_features = test_features[test_features.cp_type != 'ctl_vehicle']
    train_features = train_features.merge(train_targets_scored, on='sig_id')
    train_features = train_features.merge(train_targets_nonscored, on='sig_id')
    train_features = cate2num(train_features)
    test_features = cate2num(test_features)

    if config.normalize:
        pass    # TODO; add normalization

    # Column Infos
    score_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
    nonscore_cols = [c for c in train_targets_nonscored.columns if c not in ['sig_id']]
    cate_cols = ['cp_time', 'cp_dose'] if config.remove_vehicle else ['cp_type', 'cp_time', 'cp_dose']
    g_cols = [c for c in test_features.columns if 'g-' in c]
    c_cols = [c for c in test_features.columns if 'c-' in c]
    num_g = len(g_cols)
    num_c = len(c_cols)
    num_cate = len(cate_cols)
    input_cols = g_cols + c_cols
    if config.cat_emb_dim > 0:  # cat_emb_dim이 0보다 크면 모델에서 cate col 사용
        input_cols = cate_cols + input_cols
    X_test = test_features.loc[:, input_cols].values

    device = 'cuda' if len(config.device) > 0 else False

    test_cv_preds = []
    oof_preds = []
    oof_targets = []
    scores = []
    for seed in config.seed:
        seed_everything(seed)
        for fold in range(config.n_folds):
            print(f'[[ SEED {seed} Fold {fold} ]]')
            num_epochs = config.n_epochs
            fold_train = train_features[train_features.fold != fold].reset_index(drop=True)
            fold_valid = train_features[train_features.fold == fold].reset_index(drop=True)

            # pretrained with non-scored data
            X_train, y_train = fold_train.loc[:, input_cols].values, fold_train.loc[:, nonscore_cols].values
            X_val, y_val = fold_valid.loc[:, input_cols].values, fold_valid.loc[:, nonscore_cols].values

            model = get_tabnet(
                input_dim=X_train.shape[-1],
                output_dim=y_train.shape[-1],
                n_d=config.n_d,
                n_a=config.n_a,
                n_steps=config.n_steps,
                gamma=config.gamma,
                remove_vehicle=config.remove_vehicle,
                cat_emb_dim=config.cat_emb_dim,     # 0으로 설정하면 categorical column 사용 안함
                n_independent=config.n_independent,
                n_shared=config.n_shared,
                virtual_batch_size=config.virtual_batch_size,
                momentum=config.momentum,
                mask_type="sparsemax",      # 다른 옵션은 아직 파악을 못했습니다.
            )
            if device:
                model = model.to(device)

            train_loader = DataLoader(
                dataset=DatasetWithLabel(X_train, y_train),
                batch_size=config.batch_size,
                collate_fn=collate_fn_train,
                shuffle=True,
            )

            optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay
            )
            scheduler = OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                   max_lr=1e-2, epochs=config.n_epochs, steps_per_epoch=len(train_loader))

            for ep in range(config.n_pretrain_epochs):
                for loss in train_epoch(model, train_loader, optimizer, scheduler, config.lambda_sparse, device):
                    print(f"Epoch {ep:03d}/{config.n_pretrain_epochs:03d}", end="\r")

            X_train, y_train = fold_train.loc[:, input_cols].values, fold_train.loc[:, score_cols].values
            X_val, y_val = fold_valid.loc[:, input_cols].values, fold_valid.loc[:, score_cols].values
            in_ft = model.tabnet.final_mapping.in_features
            model.tabnet.final_mapping = nn.Linear(in_ft, y_train.shape[-1]).cuda()

            optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay
            )
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", patience=5, min_lr=1e-5, factor=0.9,
            )

            train_loader = DataLoader(
                dataset=DatasetWithLabel(X_train, y_train),
                batch_size=config.batch_size,
                collate_fn=collate_fn_train,
                shuffle=True,
            )
            val_loader = DataLoader(
                dataset=DatasetWithLabel(X_val, y_val),
                batch_size=config.batch_size,
                collate_fn=collate_fn_train,
                shuffle=False,
            )

            filename = f"{config.name}_seed{seed}_fold{fold}.pt"
            filepath = os.path.join(config.log_dir, filename)
            best_logit_log_loss = np.inf
            best_auc = -1
            logger = init_logger(config.log_dir, f'train_seed{seed}_fold{fold}.log')
            for ep in range(num_epochs):
                loss_history = []
                for loss in train_epoch(model, train_loader, optimizer, scheduler, config.lambda_sparse, device):
                    loss_history.append(loss)
                logger.info(f"Epoch {ep:03d}/{num_epochs:03d}, train loss: {np.mean(loss_history):.6f}")

                ### Predict on validation ###
                logit_log_loss, auc = validation(model, val_loader, device)
                if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    scheduler.step(logit_log_loss)
                if logit_log_loss < best_logit_log_loss:
                    best_logit_log_loss = logit_log_loss
                    # best_auc = auc
                    write_this = {
                        'model': model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'sched': scheduler.state_dict(),
                        'epoch': ep,
                    }
                    torch.save(write_this, filepath)
                    logger.info(f" ** Updated the best weight, logit log loss: {logit_log_loss:.6f}, auc: {auc:.6f} **")
                else:
                    logger.info(f"Passed to save the weight, best: {best_logit_log_loss:.6f} / logit log loss: {logit_log_loss:.6f}, auc: {auc:.6f}")

            ### Save OOF for CV ###
            best_state_dict = torch.load(filepath)
            model.load_state_dict(best_state_dict['model'])
            val_loader = DataLoader(
                DatasetWithoutLabel(X_val),
                batch_size=config.batch_size,
                collate_fn=collate_fn_test,
                shuffle=False,
            )
            preds_val = predict(model, val_loader, device)
            oof_preds.append(preds_val)
            oof_targets.append(y_val)
            scores.append(best_logit_log_loss)

            ### Predict on test ###
            test_loader = DataLoader(
                DatasetWithoutLabel(X_test),
                batch_size=config.batch_size,
                collate_fn=collate_fn_test,
                shuffle=False,
            )
            preds_test = predict(model, test_loader, device)
            test_cv_preds.append(preds_test)

    oof_preds_all = np.concatenate(oof_preds)
    oof_targets_all = np.concatenate(oof_targets)
    test_preds_all = np.stack(test_cv_preds)

    cv_score = np.mean(scores)
    all_feat = [col for col in submission.columns if col not in ["sig_id"]]
    # To obtain the same length of test_preds_all and submission
    sig_id = test_features[test_features["cp_type"] != "ctl_vehicle"].sig_id.reset_index(drop=True)
    tmp = pd.DataFrame(test_preds_all.mean(axis=0), columns=all_feat)
    tmp["sig_id"] = sig_id

    sub = pd.merge(test_features[["sig_id"]], tmp, on="sig_id", how="left")
    sub.fillna(0.001, inplace=True)
    sub.to_csv(os.path.join(config.sub_dir, f'{config.name}_{cv_score:.6f}.csv'), index=False)

    main_logger.info(f'[cv:{cv_score:.6f}] {config.name} Finished')

    cv_seeds = [scores[i:i + config.n_folds] for i in range(0, len(scores), config.n_folds)]
    for idx in range(len(cv_seeds)):
        main_logger.info(f'SEED {config.seed[idx]} CV score: {np.mean(cv_seeds[idx]):.6f}')
