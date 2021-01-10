import torch
from torch.utils.data import Dataset

from dataset.preprocess import cate2num


class moaDataset(Dataset):
    def __init__(self, sig_ids, g_values, c_values, cate_values,
                 labels, non_labels=None, test=False):
        self.sig_ids = sig_ids
        self.g_values = g_values
        self.c_values = c_values
        self.cate_values = cate_values
        self.labels = labels
        self.non_labels = non_labels
        self.test = test

    def __len__(self):
        return len(self.sig_ids)

    def __getitem__(self, idx):
        g_x = torch.FloatTensor(self.g_values[idx])
        c_x = torch.FloatTensor(self.c_values[idx])
        cate_x = torch.FloatTensor(self.cate_values[idx])

        if self.test:
            return g_x, c_x, cate_x
        else:
            label = torch.tensor(self.labels[idx]).float()
            non_label = torch.tensor(self.non_labels[idx]).float()
            return g_x, c_x, cate_x, label, non_label


class DatasetWithLabel(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self._X = X
        self._y = y

    def __getitem__(self, item):
        return self._X[item], self._y[item]

    def __len__(self):
        return len(self._y)


def collate_fn_train(batch):
    X = [x[0] for x in batch]
    X = torch.FloatTensor(X)

    y = [x[1] for x in batch]
    y = torch.FloatTensor(y)
    return X, y


class DatasetWithoutLabel(Dataset):
    def __init__(self, X):
        super().__init__()
        self._X = X

    def __getitem__(self, item):
        return (self._X[item], )

    def __len__(self):
        return len(self._X)


def collate_fn_test(batch):
    X = [x[0] for x in batch]
    X = torch.FloatTensor(X)
    return X
