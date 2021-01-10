import os
from shutil import copytree, ignore_patterns

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from configs.random_search import RandomParams
from configs.tabnet import config as conf
from TabNet_Train import get_tabnet, train_epoch, validation, predict
from dataset.moaDataset import cate2num
from dataset.moaDataset import DatasetWithLabel, DatasetWithoutLabel, collate_fn_train, collate_fn_test
from scripts.utils import seed_everything


rp = RandomParams()
rp.set('n_d', [8, 16, 32, 64], 'choice')
rp.set('n_independent', [1, 2, 3], 'choice')
rp.set('n_shared', [0, 1, 2, 3], 'choice')
rp.set('cat_emb_dim', [0, 4, 8], 'choice')
rp.set('n_steps', [3, 4, 5, 6, 7], 'choice')
rp.set('gamma', [1.0, 2.0], 'range')
# rp.set('batch_size', [128, 256, 512, 1024], 'choice')
# rp.set('weight_decay', [0.0, 1e-5], 'choice')


NUM_SEARCH = 20


if __name__ == "__main__":
    config = conf()

    # Backup Code
    dst_dir = os.path.join(config.log_dir, 'code')
    if not os.path.isdir(dst_dir):
        copytree(config.bkup_dir, dst_dir, ignore=ignore_patterns('.*'))

    # Load Data
    train_features = pd.read_csv(f'{config.data_dir}/train_features_RSGKF_combined.csv')
    train_targets_scored = pd.read_csv(f'{config.data_dir}/train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv(f'{config.data_dir}/train_targets_nonscored.csv')

    # Preprocess Data
    if config.remove_vehicle:
        train_features = train_features[train_features.cp_type != 'ctl_vehicle']
    train_features = cate2num(train_features)

    # Column Info
    score_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
    nonscore_cols = [c for c in train_targets_nonscored.columns if c not in ['sig_id']]
    cate_cols = ['cp_time', 'cp_dose'] if config.remove_vehicle else ['cp_type', 'cp_time', 'cp_dose']
    g_cols = [c for c in train_features.columns if c.startswith('g-')]
    c_cols = [c for c in train_features.columns if c.startswith('c-')]
    ae_cols = [c for c in train_features.columns if c.startswith('ae-')]
    input_cols = g_cols + c_cols
    if config.cat_emb_dim > 0:  # cat_emb_dim이 0보다 크면 모델에서 cate col 사용
        input_cols = cate_cols + input_cols
    if config.use_autoencoder:
        input_cols = input_cols + ae_cols

    if config.normalize:
        pass    # TODO; add normalization

    is_cuda = config.device == 'cuda'

    oof_preds = []
    oof_targets = []
    scores = []

    seed = config.seed[0]   # use one seed
    seed_everything(seed)

    for search_iter in range(NUM_SEARCH):
        # change
        rp(config)

        for fold in [0]:
        # for fold in range(config.n_folds):
            print(f'[[ ITER {search_iter} Fold {fold} ]]')
            fold_train = train_features[train_features.fold != fold].reset_index(drop=True)
            fold_valid = train_features[train_features.fold == fold].reset_index(drop=True)

            input_features = []
            if config.cat_emb_dim > 0:  # cat_emb_dim이 0보다 크면 모델에서 cate col 사용; 항상 제일 앞에 와야함
                cate_train = fold_train.loc[:, cate_cols].values
                input_features.append(cate_train)

            g_train = fold_train.loc[:, g_cols].values
            if config.num_pca_g > 0:
                pca_g = PCA(n_components=config.num_pca_g, random_state=seed)
                pca_g.fit(g_train)
                g_train = pca_g.transform(g_train)
            input_features.append(g_train)

            c_train = fold_train.loc[:, c_cols].values
            if config.num_pca_c > 0:
                pca_c = PCA(n_components=config.num_pca_c, random_state=seed)
                pca_c.fit(c_train)
                c_train = pca_c.transform(c_train)
            input_features.append(c_train)

            if config.use_autoencoder:
                ae_train = fold_train.loc[:, ae_cols].values
                input_features.append(ae_train)

            X_train = np.concatenate(input_features, axis=-1)
            y_train = fold_train.loc[:, score_cols].values

            # valid set
            input_features = []
            if config.cat_emb_dim > 0:  # cat_emb_dim이 0보다 크면 모델에서 cate col 사용; 항상 제일 앞에 와야함
                cate_valid = fold_valid.loc[:, cate_cols].values
                input_features.append(cate_valid)

            g_valid = fold_valid.loc[:, g_cols].values
            if config.num_pca_g > 0:
                g_valid = pca_g.transform(g_valid)
            input_features.append(g_valid)

            c_valid = fold_valid.loc[:, c_cols].values
            if config.num_pca_c > 0:
                c_valid = pca_c.transform(c_valid)
            input_features.append(c_valid)

            if config.use_autoencoder:
                ae_valid = fold_valid.loc[:, ae_cols].values
                input_features.append(ae_valid)

            X_valid = np.concatenate(input_features, axis=-1)
            y_valid = fold_valid.loc[:, score_cols].values

            train_loader = DataLoader(
                dataset=DatasetWithLabel(X_train, y_train),
                batch_size=config.batch_size,
                collate_fn=collate_fn_train,
                shuffle=True,
                pin_memory=True,
            )
            val_loader = DataLoader(
                dataset=DatasetWithLabel(X_valid, y_valid),
                batch_size=config.batch_size,
                collate_fn=collate_fn_train,
                shuffle=False,
                pin_memory=True,
            )

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
                # mask_type="sparsemax",      # 다른 옵션은 아직 파악을 못했습니다.
                mask_type="entmax",
            )
            if is_cuda:
                model = model.cuda()

            ### Fit ###
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay
            )
            # TODO; scheduler type in config
            scheduler = StepLR(
                optimizer,
                step_size=config.scheduler_step_size,
                gamma=config.scheduler_gamma,
            )
            num_epochs = config.n_epochs or int(config.n_train_steps / len(train_loader))

            filename = f"{config.name}_iter{search_iter}_fold{fold}.pt"
            filepath = os.path.join(config.log_dir, filename)
            best_logit_log_loss = np.inf
            best_auc = -1

            stopper = 0
            for ep in range(num_epochs):
                loss_history = []
                for loss in train_epoch(model, train_loader, optimizer, scheduler, config.lambda_sparse, is_cuda):
                    loss_history.append(loss)
                print(f"Epoch {ep:03d}/{num_epochs:03d}, train loss: {np.mean(loss_history):.6f}")

                ### Predict on validation ###
                # TODO; AUC 기준?
                logit_log_loss, auc = validation(model, val_loader, is_cuda)
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
                    stopper = 0
                    print(f" ** Updated the best weight, logit log loss: {logit_log_loss:.6f}, auc: {auc:.6f} **")
                else:
                    stopper += 1
                    print(f"Passed to save the weight, best: {best_logit_log_loss:.6f} / logit log loss: {logit_log_loss:.6f}, auc: {auc:.6f}")
                    if stopper > 20:    # patience
                        print("Early stopped.")
                        break

            ### Save OOF for CV ###
            best_state_dict = torch.load(filepath)
            model.load_state_dict(best_state_dict['model'])
            val_loader = DataLoader(
                DatasetWithoutLabel(X_valid),
                batch_size=config.batch_size,
                collate_fn=collate_fn_test,
                shuffle=False,
                pin_memory=True,
            )
            preds_val = predict(model, val_loader, is_cuda)
            oof_preds.append(preds_val)
            oof_targets.append(y_valid)
            scores.append(best_logit_log_loss)

    oof_preds_all = np.concatenate(oof_preds)
    oof_targets_all = np.concatenate(oof_targets)

    cv_iter = [scores[i:i + config.n_folds] for i in range(0, len(scores), NUM_SEARCH)]
    for idx in range(len(cv_iter)):
        print(f'ITER {idx} CV score: {np.mean(cv_iter[idx]):.6f}')
