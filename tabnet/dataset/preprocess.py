import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from scipy.stats import kurtosis

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA, NMF
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def cate2num(df):
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72: 2})
    df['cp_dose'] = df['cp_dose'].map({'D1': 0, 'D2': 1})
    return df


def get_cols(test):
    g_cols = [c for c in test.columns if 'g-' in c]
    c_cols = [c for c in test.columns if 'c-' in c]
    num_g = len(g_cols)
    num_c = len(c_cols)
    return g_cols, c_cols, num_g, num_c


def make_folds(train_features, train_targets_scored, n_folds, seed):
    train = train_features.merge(train_targets_scored, on='sig_id')
    target_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
    cols = target_cols + ['cp_type']

    train_cp = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_cp.loc[:, 'fold'] = 0
    for n, (train_index, val_index) in enumerate(mskf.split(train_cp, train_cp[target_cols])):
        train_cp.loc[val_index, 'fold'] = int(n)
    train_cp['fold'] = train_cp['fold'].astype(int)

    train_ctl = train[train['cp_type'] == 'ctl_vehicle'].reset_index(drop=True)
    train_ctl.loc[:, 'fold'] = 100

    train = pd.concat([train_cp, train_ctl])
    train_features_with_fold = train_features.merge(train[['sig_id', 'fold']], on='sig_id')
    return train_features_with_fold


# https://www.kaggle.com/vbmokin/moa-pytorch-rankgauss-pca-nn-upgrade-3d-visual?scriptVersionId=46264255
def create_rankgauss(train, test, cols, QT_n_quantile_min, QT_n_quantile_max, seed):
    for col in (cols):
        kurt = max(kurtosis(train[col]), kurtosis(test[col]))
        QuantileTransformer_n_quantiles = n_quantile_for_kurt(kurt, calc_QT_par_kurt(QT_n_quantile_min,
                                                                                     QT_n_quantile_max))
        transformer = QuantileTransformer(n_quantiles=QuantileTransformer_n_quantiles, random_state=seed,
                                          output_distribution="normal")
        vec_len = len(train[col].values)
        vec_len_test = len(test[col].values)
        raw_vec = train[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)

        train[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        test[col] = \
        transformer.transform(test[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]
        return train, test


def calc_QT_par_kurt(QT_n_quantile_min=10, QT_n_quantile_max=200):
    # Calculation parameters of function: n_quantile(kurtosis) = k1*kurtosis + k0
    # For Train & Test datasets (GENES + CELLS features): minimum kurtosis = 1.53655, maximum kurtosis = 30.4929

    a = np.array([[1.53655, 1], [30.4929, 1]])
    b = np.array([QT_n_quantile_min, QT_n_quantile_max])

    return np.linalg.solve(a, b)


def n_quantile_for_kurt(kurt, calc_QT_par_kurt_transform):
    # Calculation parameters of function: n_quantile(kurtosis) = calc_QT_par_kurt_transform[0]*kurtosis + calc_QT_par_kurt_transform[1]
    return int(calc_QT_par_kurt_transform[0] * kurt + calc_QT_par_kurt_transform[1])


# Function to extract common stats features
def fe_stats(train, test, cols, kind='g'):
    for df in [train, test]:
        df[f'{kind}_sum'] = df[cols].sum(axis = 1)
        df[f'{kind}_mean'] = df[cols].mean(axis = 1)
        df[f'{kind}_std'] = df[cols].std(axis = 1)
        df[f'{kind}_kurt'] = df[cols].kurtosis(axis = 1)
        df[f'{kind}_skew'] = df[cols].skew(axis = 1)
    return train, test


def create_pca(train, test, cols, seed, kind='g', n_components=50, drop=False):
    train_ = train[cols].copy()
    test_ = test[cols].copy()
    data = pd.concat([train_, test_], axis = 0)
    pca = PCA(n_components=n_components, random_state=seed)
    data = pca.fit_transform(data)
    columns = [f'{kind}-pca-{i}' for i in range(n_components)]
    data = pd.DataFrame(data, columns = columns)
    train_ = data.iloc[:train.shape[0]]
    test_ = data.iloc[train.shape[0]:].reset_index(drop=True)
    if drop:
        train = train.drop(columns=cols)
        test = test.drop(columns=cols)
    train = pd.concat([train.reset_index(drop = True), train_], axis = 1)
    test = pd.concat([test.reset_index(drop = True), test_], axis = 1)
    return train, test


def fe_squared(train, test, cols):
    for df in [train, test]:
        for col in cols:
            df[f'{col}_squared'] = df[col] ** 2
    return train, test


# Function to scale our data
def scaling(train, test, cols):
    scaler = RobustScaler()
    scaler.fit(pd.concat([train[cols], test[cols]], axis = 0))
    train[cols] = scaler.transform(train[cols])
    test[cols] = scaler.transform(test[cols])
    return train, test


def vt_selection(train, test, cols, VarianceThreshold_for_FS, kind='g', drop=True):
    train_ = train[cols].copy()
    test_ = test[cols].copy()
    var_thresh = VarianceThreshold(VarianceThreshold_for_FS)

    data = train_.append(test_)
    data_transformed = var_thresh.fit_transform(data)

    train_features_transformed = data_transformed[ : train_.shape[0]]
    test_features_transformed = data_transformed[-test_.shape[0] : ]
    columns = [f'{kind}-vt-{i}' for i in range(train_features_transformed.shape[1])]
    train_features_transformed = pd.DataFrame(train_features_transformed, columns = columns)
    test_features_transformed = pd.DataFrame(test_features_transformed, columns = columns)
    if drop:
        train = train.drop(columns=cols)
        test = test.drop(columns=cols)
    train = pd.concat([train.reset_index(drop=True), train_features_transformed], axis = 1)
    test = pd.concat([test.reset_index(drop=True), test_features_transformed], axis = 1)
    return train, test


def lsvc_selection(train, test, cols, score_cols, kind='g', model='lsvc', drop=True):
    train_ = train[cols].copy()
    test_ = test[cols].copy()

    if model == 'lsvc':
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_.values, [''.join(map(str, elem)) for elem in
                                                                               train[score_cols].values])
    else:
        raise NotImplementedError
    model = SelectFromModel(lsvc, prefit=True)

    train_ = model.transform(train_[cols].values)
    test_ = model.transform(test_[cols].values)
    columns = [f'{kind}-lsvc-{i}' for i in range(train_.shape[1])]
    train_ = pd.DataFrame(train_, columns=columns)
    test_ = pd.DataFrame(test_, columns=columns)
    if drop:
        train = train.drop(columns=cols)
        test = test.drop(columns=cols)
    train = pd.concat([train.reset_index(drop=True), train_], axis=1)
    test = pd.concat([test.reset_index(drop=True), test_], axis=1)
    return train, test


def preprocess_fn(train_features, test_features, train_targets_scored, config):
    score_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
    train_features = make_folds(train_features, train_targets_scored, config.n_folds, config.seed[0])

    if config.rank_gauss:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = create_rankgauss(train_features, test_features, g_cols + c_cols,
                                                         config.QT_n_quantile_min, config.QT_n_quantile_max,
                                                         seed=config.seed[0])

    if config.feature_status:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = fe_stats(train_features, test_features, g_cols, kind='g')
        train_features, test_features = fe_stats(train_features, test_features, c_cols, kind='c')

    if config.pca:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = create_pca(train_features, test_features, g_cols,
                                                   seed=config.seed[0],
                                                   kind='g',
                                                   n_components=config.n_comp_GENES)
        train_features, test_features = create_pca(train_features, test_features, c_cols,
                                                   seed=config.seed[0],
                                                   kind='c',
                                                   n_components=config.n_comp_CELLS)

    if config.cell_squared:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = fe_squared(train_features, test_features, c_cols)

    if config.scaling:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = scaling(train_features, test_features, g_cols)
        train_features, test_features = scaling(train_features, test_features, c_cols)

    if config.vt_selection:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = vt_selection(train_features, test_features, g_cols,
                                                     config.VarianceThreshold_for_FS, kind='g', drop=True)
        train_features, test_features = vt_selection(train_features, test_features, c_cols,
                                                     config.VarianceThreshold_for_FS, kind='c', drop=True)

    if config.lsvc_selection:
        g_cols, c_cols, num_g, num_c = get_cols(test_features)
        train_features, test_features = lsvc_selection(train_features, test_features, g_cols, score_cols, kind='g')
        train_features, test_features = lsvc_selection(train_features, test_features, c_cols, score_cols, kind='c')

    return train_features, test_features