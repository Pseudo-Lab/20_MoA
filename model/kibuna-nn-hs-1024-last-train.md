ref: https://www.kaggle.com/kibuna/kibuna-nn-hs-1024-last-train/data

- a notebook to save preprocessing model and train/save NN models
- all necessary ouputs are stored in MODEL_DIR = output/kaggle/working/model
    - put those into dataset, and load it from inference notebook


```python
import sys

%mkdir model
%mkdir interim

from umap import UMAP
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns
import time

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,FactorAnalysis
from sklearn.manifold import TSNE

from scipy.sparse.csgraph import connected_components
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
print(torch.cuda.is_available())
import warnings
# warnings.filterwarnings('ignore')
```

    mkdir: cannot create directory ‘model’: File exists
    mkdir: cannot create directory ‘interim’: File exists
    True



```python
torch.__version__
```




    '1.7.1'




```python
NB = '25'

IS_TRAIN = True
MODEL_DIR = "model" # "../model"
INT_DIR = "interim" # "../interim"

NSEEDS = 5  # 5
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 15
BATCH_SIZE = 256
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False

NFOLDS = 5  # 5

PMIN = 0.0005
PMAX = 0.9995
SMIN = 0.0
SMAX = 1.0
```


```python
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
```


```python
train_targets_nonscored = train_targets_nonscored.loc[:, train_targets_nonscored.sum() != 0]
print(train_targets_nonscored.shape)
```

    (23814, 332)



```python
# for c in train_targets_scored.columns:
#     if c != "sig_id":
#         train_targets_scored[c] = np.maximum(PMIN, np.minimum(PMAX, train_targets_scored[c]))
for c in train_targets_nonscored.columns:
    if c != "sig_id":
        train_targets_nonscored[c] = np.maximum(PMIN, np.minimum(PMAX, train_targets_nonscored[c]))
```


```python
print("(nsamples, nfeatures)")
print(train_features.shape)
print(train_targets_scored.shape)
print(train_targets_nonscored.shape)
print(test_features.shape)
print(sample_submission.shape)
```

    (nsamples, nfeatures)
    (23814, 876)
    (23814, 207)
    (23814, 332)
    (3982, 876)
    (3982, 207)



```python
GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
```


```python
def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=1903)
```


```python

```


```python
# GENES
n_comp = 90
n_dim = 45

data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])

if IS_TRAIN:
    fa = FactorAnalysis(n_components=n_comp, random_state=1903).fit(data[GENES])
    pd.to_pickle(fa, f'{MODEL_DIR}/{NB}_factor_analysis_g.pkl')
    umap = UMAP(n_components=n_dim, random_state=1903).fit(data[GENES])
    pd.to_pickle(umap, f'{MODEL_DIR}/{NB}_umap_g.pkl')
else:
    fa = pd.read_pickle(f'{MODEL_DIR}/{NB}_factor_analysis_g.pkl')
    umap = pd.read_pickle(f'{MODEL_DIR}/{NB}_umap_g.pkl')

data2 = (fa.transform(data[GENES]))
data3 = (umap.transform(data[GENES]))

train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]
train3 = data3[:train_features.shape[0]]
test3 = data3[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'fa_G-{i}' for i in range(n_comp)])
train3 = pd.DataFrame(train3, columns=[f'umap_G-{i}' for i in range(n_dim)])
test2 = pd.DataFrame(test2, columns=[f'fa_G-{i}' for i in range(n_comp)])
test3 = pd.DataFrame(test3, columns=[f'umap_G-{i}' for i in range(n_dim)])

train_features = pd.concat((train_features, train2, train3), axis=1)
test_features = pd.concat((test_features, test2, test3), axis=1)

#CELLS
n_comp = 50
n_dim = 25

data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])

if IS_TRAIN:
    fa = FactorAnalysis(n_components=n_comp, random_state=1903).fit(data[CELLS])
    pd.to_pickle(fa, f'{MODEL_DIR}/{NB}_factor_analysis_c.pkl')
    umap = UMAP(n_components=n_dim, random_state=1903).fit(data[CELLS])
    pd.to_pickle(umap, f'{MODEL_DIR}/{NB}_umap_c.pkl')
else:
    fa = pd.read_pickle(f'{MODEL_DIR}/{NB}_factor_analysis_c.pkl')
    umap = pd.read_pickle(f'{MODEL_DIR}/{NB}_umap_c.pkl')
    
data2 = (fa.transform(data[CELLS]))
data3 = (umap.transform(data[CELLS]))

train2 = data2[:train_features.shape[0]]
test2 = data2[-test_features.shape[0]:]
train3 = data3[:train_features.shape[0]]
test3 = data3[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'fa_C-{i}' for i in range(n_comp)])
train3 = pd.DataFrame(train3, columns=[f'umap_C-{i}' for i in range(n_dim)])
test2 = pd.DataFrame(test2, columns=[f'fa_C-{i}' for i in range(n_comp)])
test3 = pd.DataFrame(test3, columns=[f'umap_C-{i}' for i in range(n_dim)])

train_features = pd.concat((train_features, train2, train3), axis=1)
test_features = pd.concat((test_features, test2, test3), axis=1)

# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
```


```python

```


```python
from sklearn.preprocessing import QuantileTransformer

for col in (GENES + CELLS):
    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)
    raw_vec = pd.concat([train_features, test_features])[col].values.reshape(vec_len+vec_len_test, 1)
    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=123, output_distribution="normal")
        transformer.fit(raw_vec)
        pd.to_pickle(transformer, f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')
    else:
        transformer = pd.read_pickle(f'{MODEL_DIR}/{NB}_{col}_quantile_transformer.pkl')        

    train_features[col] = transformer.transform(train_features[col].values.reshape(vec_len, 1)).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]
```


```python
# PCAS = [col for col in train_features.columns if col.startswith('pca_')]
# UMAPS = [col for col in train_features.columns if col.startswith('umap_')]
```


```python
# from sklearn.preprocessing import PolynomialFeatures
# n_deg = 2

# data = pd.concat([pd.DataFrame(train_features[PCAS]), pd.DataFrame(test_features[PCAS])])
# data2 = (PolynomialFeatures(degree=n_deg, include_bias=False).fit_transform(data[PCAS]))

# # print(data2)
# # data4 = (UMAP(n_components=n_dim, n_neighbors=5, random_state=1903).fit_transform(data[GENES]))
# # data5 = (UMAP(n_components=n_dim, min_dist=0.01, random_state=1903).fit_transform(data[GENES]))

# train2 = data2[:train_features.shape[0]]
# test2 = data2[-test_features.shape[0]:]

# # print(train2.shape)
# train2 = pd.DataFrame(train2, columns=[f'poly_C-{i}' for i in range(train2.shape[1])])
# test2 = pd.DataFrame(test2, columns=[f'poly_C-{i}' for i in range(train2.shape[1])])

# # drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
# # train_features = pd.concat((train_features, train2, train3, train4, train5), axis=1)
# # test_features = pd.concat((test_features, test2, test3, test4, test5), axis=1)
# train_features = pd.concat((train_features, train2), axis=1)
# test_features = pd.concat((test_features, test2), axis=1)


# data = pd.concat([pd.DataFrame(train_features[UMAPS]), pd.DataFrame(test_features[UMAPS])])
# data2 = (PolynomialFeatures(degree=n_deg, include_bias=False).fit_transform(data[UMAPS]))

# # print(data2)
# # data4 = (UMAP(n_components=n_dim, n_neighbors=5, random_state=1903).fit_transform(data[GENES]))
# # data5 = (UMAP(n_components=n_dim, min_dist=0.01, random_state=1903).fit_transform(data[GENES]))

# train2 = data2[:train_features.shape[0]]
# test2 = data2[-test_features.shape[0]:]

# # print(train2.shape)
# train2 = pd.DataFrame(train2, columns=[f'poly_C-{i}' for i in range(train2.shape[1])])
# test2 = pd.DataFrame(test2, columns=[f'poly_C-{i}' for i in range(train2.shape[1])])

# # drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
# # train_features = pd.concat((train_features, train2, train3, train4, train5), axis=1)
# # test_features = pd.concat((test_features, test2, test3, test4, test5), axis=1)
# train_features = pd.concat((train_features, train2), axis=1)
# test_features = pd.concat((test_features, test2), axis=1)
```


```python
print(train_features.shape)
print(test_features.shape)
```

    (23814, 1086)
    (3982, 1086)



```python

```


```python
# train = train_features.merge(train_targets_scored, on='sig_id')
train = train_features.merge(train_targets_nonscored, on='sig_id')
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

# target = train[train_targets_scored.columns]
target = train[train_targets_nonscored.columns]
```


```python
train = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)
```


```python
print(target.shape)
print(train_features.shape)
print(test_features.shape)
print(train.shape)
print(test.shape)
```

    (21948, 332)
    (23814, 1086)
    (3982, 1086)
    (21948, 1416)
    (3624, 1085)



```python
target_cols = target.drop('sig_id', axis=1).columns.values.tolist()
```


```python
folds = train.copy()

mskf = MultilabelStratifiedKFold(n_splits=NFOLDS)

for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
    folds.loc[v_idx, 'kfold'] = int(f)

folds['kfold'] = folds['kfold'].astype(int)
folds
```

    /opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py:70: FutureWarning: Pass shuffle=False, random_state=None as keyword args. From version 0.25 passing these as positional arguments will result in an error
      FutureWarning)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>g-0</th>
      <th>g-1</th>
      <th>g-2</th>
      <th>g-3</th>
      <th>g-4</th>
      <th>g-5</th>
      <th>g-6</th>
      <th>...</th>
      <th>vasopressin_receptor_antagonist</th>
      <th>ve-cadherin_antagonist</th>
      <th>vesicular_monoamine_transporter_inhibitor</th>
      <th>vitamin_k_antagonist</th>
      <th>voltage-gated_potassium_channel_activator</th>
      <th>voltage-gated_sodium_channel_blocker</th>
      <th>wdr5_mll_interaction_inhibitor</th>
      <th>xanthine_oxidase_inhibitor</th>
      <th>xiap_inhibitor</th>
      <th>kfold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>24</td>
      <td>D1</td>
      <td>1.146806</td>
      <td>0.902075</td>
      <td>-0.418339</td>
      <td>-0.961202</td>
      <td>-0.254770</td>
      <td>-1.021300</td>
      <td>-1.369236</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>72</td>
      <td>D1</td>
      <td>0.128824</td>
      <td>0.676862</td>
      <td>0.274345</td>
      <td>0.090495</td>
      <td>1.208863</td>
      <td>0.688965</td>
      <td>0.316734</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_000a6266a</td>
      <td>48</td>
      <td>D1</td>
      <td>0.790372</td>
      <td>0.939951</td>
      <td>1.428097</td>
      <td>-0.121817</td>
      <td>-0.002067</td>
      <td>1.495091</td>
      <td>0.238763</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_0015fd391</td>
      <td>48</td>
      <td>D1</td>
      <td>-0.729866</td>
      <td>-0.277163</td>
      <td>-0.441200</td>
      <td>0.766612</td>
      <td>2.347817</td>
      <td>-0.862761</td>
      <td>-2.308829</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_001626bd3</td>
      <td>72</td>
      <td>D2</td>
      <td>-0.444558</td>
      <td>-0.481202</td>
      <td>0.974729</td>
      <td>0.977467</td>
      <td>1.468304</td>
      <td>-0.874772</td>
      <td>-0.372682</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21943</th>
      <td>id_fff8c2444</td>
      <td>72</td>
      <td>D1</td>
      <td>0.247623</td>
      <td>-1.231184</td>
      <td>0.221572</td>
      <td>-0.354096</td>
      <td>-0.332073</td>
      <td>0.570635</td>
      <td>-0.150125</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21944</th>
      <td>id_fffb1ceed</td>
      <td>24</td>
      <td>D2</td>
      <td>0.217613</td>
      <td>-0.027031</td>
      <td>-0.237430</td>
      <td>-0.787215</td>
      <td>-0.677817</td>
      <td>0.919474</td>
      <td>0.742866</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>2</td>
    </tr>
    <tr>
      <th>21945</th>
      <td>id_fffb70c0c</td>
      <td>24</td>
      <td>D2</td>
      <td>-1.914666</td>
      <td>0.581880</td>
      <td>-0.588706</td>
      <td>1.303439</td>
      <td>-1.009079</td>
      <td>0.852202</td>
      <td>-0.302814</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21946</th>
      <td>id_fffcb9e7c</td>
      <td>24</td>
      <td>D1</td>
      <td>0.826302</td>
      <td>0.411235</td>
      <td>0.433297</td>
      <td>0.307575</td>
      <td>1.075324</td>
      <td>-0.024425</td>
      <td>0.051483</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>3</td>
    </tr>
    <tr>
      <th>21947</th>
      <td>id_ffffdd77b</td>
      <td>72</td>
      <td>D1</td>
      <td>-1.245739</td>
      <td>1.567230</td>
      <td>-0.269829</td>
      <td>1.092958</td>
      <td>-0.515819</td>
      <td>-2.091765</td>
      <td>-1.627645</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>21948 rows × 1417 columns</p>
</div>




```python
print(train.shape)
print(folds.shape)
print(test.shape)
print(target.shape)
print(sample_submission.shape)
```

    (21948, 1416)
    (21948, 1417)
    (3624, 1085)
    (21948, 332)
    (3982, 207)



```python
class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
        }
        return dct
    
class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct
```


```python
def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
#         print(inputs.shape)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
        
        preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds
```


```python
class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.15)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x
```


```python
def process_data(data):
    
    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
#     data.loc[:, 'cp_time'] = data.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})
#     data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

# --------------------- Normalize ---------------------
#     for col in GENES:
#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))
    
#     for col in CELLS:
#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))
    
#--------------------- Removing Skewness ---------------------
#     for col in GENES + CELLS:
#         if(abs(data[col].skew()) > 0.75):
            
#             if(data[col].skew() < 0): # neg-skewness
#                 data[col] = data[col].max() - data[col] + 1
#                 data[col] = np.sqrt(data[col])
            
#             else:
#                 data[col] = np.sqrt(data[col])
    
    return data
```


```python
feature_cols = [c for c in process_data(folds).columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]
len(feature_cols)
```




    1087




```python
num_features=len(feature_cols)
num_targets=len(target_cols)
hidden_size=2048
# hidden_size=4096
# hidden_size=9192
```


```python
def run_training(fold, seed):
    
    seed_everything(seed)
    
    train = process_data(folds)
    test_ = process_data(test)
    
    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index
    
    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)
    
    x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values
    
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.3, div_factor=1000, 
#                                               max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.2, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    loss_fn = nn.BCEWithLogitsLoss()
    
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
    
    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    best_loss = np.inf
    best_loss_epoch = -1
    
    if IS_TRAIN:
        for epoch in range(EPOCHS):

            train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)

            if valid_loss < best_loss:            
                best_loss = valid_loss
                best_loss_epoch = epoch
                oof[val_idx] = valid_preds
                torch.save(model.state_dict(), f"{MODEL_DIR}/{NB}-nonscored-SEED{seed}-FOLD{fold}_.pth")

            elif(EARLY_STOP == True):
                early_step += 1
                if (early_step >= early_stopping_steps):
                    break

            if epoch % 10 == 0 or epoch == EPOCHS-1:
                print(f"seed: {seed}, FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}, best_loss: {best_loss:.6f}, best_loss_epoch: {best_loss_epoch}")            
    
    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.load_state_dict(torch.load(f"{MODEL_DIR}/{NB}-nonscored-SEED{seed}-FOLD{fold}_.pth"))
    model.to(DEVICE)
    
    if not IS_TRAIN:
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
        oof[val_idx] = valid_preds    
    
    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)
    
    return oof, predictions
```


```python
def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))
    
    for fold in range(NFOLDS):
        oof_, pred_ = run_training(fold, seed)
        
        predictions += pred_ / NFOLDS
        oof += oof_
        
    return oof, predictions
```


```python
SEED = range(NSEEDS)  #[0, 1, 2, 3 ,4]#, 5, 6, 7, 8, 9, 10]
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

time_start = time.time()

for seed in SEED:
    
    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)
    print(f"elapsed time: {time.time() - time_start}")

train[target_cols] = oof
test[target_cols] = predictions

print(oof.shape)
print(predictions.shape)
```

    seed: 0, FOLD: 0, EPOCH: 0, train_loss: 0.648275, valid_loss: 0.214343, best_loss: 0.214343, best_loss_epoch: 0
    seed: 0, FOLD: 0, EPOCH: 10, train_loss: 0.008612, valid_loss: 0.008796, best_loss: 0.008796, best_loss_epoch: 10
    seed: 0, FOLD: 0, EPOCH: 14, train_loss: 0.008108, valid_loss: 0.008678, best_loss: 0.008678, best_loss_epoch: 14
    seed: 0, FOLD: 1, EPOCH: 0, train_loss: 0.647838, valid_loss: 0.215069, best_loss: 0.215069, best_loss_epoch: 0
    seed: 0, FOLD: 1, EPOCH: 10, train_loss: 0.008602, valid_loss: 0.008954, best_loss: 0.008908, best_loss_epoch: 9
    seed: 0, FOLD: 1, EPOCH: 14, train_loss: 0.008065, valid_loss: 0.008705, best_loss: 0.008705, best_loss_epoch: 14
    seed: 0, FOLD: 2, EPOCH: 0, train_loss: 0.647599, valid_loss: 0.214583, best_loss: 0.214583, best_loss_epoch: 0
    seed: 0, FOLD: 2, EPOCH: 10, train_loss: 0.008594, valid_loss: 0.008835, best_loss: 0.008835, best_loss_epoch: 10
    seed: 0, FOLD: 2, EPOCH: 14, train_loss: 0.008053, valid_loss: 0.008724, best_loss: 0.008724, best_loss_epoch: 14
    seed: 0, FOLD: 3, EPOCH: 0, train_loss: 0.646752, valid_loss: 0.231156, best_loss: 0.231156, best_loss_epoch: 0
    seed: 0, FOLD: 3, EPOCH: 10, train_loss: 0.008570, valid_loss: 0.008973, best_loss: 0.008973, best_loss_epoch: 10
    seed: 0, FOLD: 3, EPOCH: 14, train_loss: 0.008068, valid_loss: 0.008869, best_loss: 0.008869, best_loss_epoch: 14
    seed: 0, FOLD: 4, EPOCH: 0, train_loss: 0.648380, valid_loss: 0.254549, best_loss: 0.254549, best_loss_epoch: 0
    seed: 0, FOLD: 4, EPOCH: 10, train_loss: 0.008591, valid_loss: 0.008440, best_loss: 0.008427, best_loss_epoch: 9
    seed: 0, FOLD: 4, EPOCH: 14, train_loss: 0.008094, valid_loss: 0.008304, best_loss: 0.008304, best_loss_epoch: 14
    elapsed time: 108.55218839645386
    seed: 1, FOLD: 0, EPOCH: 0, train_loss: 0.648185, valid_loss: 0.217353, best_loss: 0.217353, best_loss_epoch: 0
    seed: 1, FOLD: 0, EPOCH: 10, train_loss: 0.008631, valid_loss: 0.008756, best_loss: 0.008756, best_loss_epoch: 10
    seed: 1, FOLD: 0, EPOCH: 14, train_loss: 0.008062, valid_loss: 0.008673, best_loss: 0.008673, best_loss_epoch: 14
    seed: 1, FOLD: 1, EPOCH: 0, train_loss: 0.648193, valid_loss: 0.233665, best_loss: 0.233665, best_loss_epoch: 0
    seed: 1, FOLD: 1, EPOCH: 10, train_loss: 0.008616, valid_loss: 0.008844, best_loss: 0.008844, best_loss_epoch: 10
    seed: 1, FOLD: 1, EPOCH: 14, train_loss: 0.008121, valid_loss: 0.008712, best_loss: 0.008712, best_loss_epoch: 14
    seed: 1, FOLD: 2, EPOCH: 0, train_loss: 0.647340, valid_loss: 0.217425, best_loss: 0.217425, best_loss_epoch: 0
    seed: 1, FOLD: 2, EPOCH: 10, train_loss: 0.008593, valid_loss: 0.008839, best_loss: 0.008839, best_loss_epoch: 10
    seed: 1, FOLD: 2, EPOCH: 14, train_loss: 0.008039, valid_loss: 0.008728, best_loss: 0.008728, best_loss_epoch: 14
    seed: 1, FOLD: 3, EPOCH: 0, train_loss: 0.647800, valid_loss: 0.204302, best_loss: 0.204302, best_loss_epoch: 0
    seed: 1, FOLD: 3, EPOCH: 10, train_loss: 0.008514, valid_loss: 0.008948, best_loss: 0.008948, best_loss_epoch: 10
    seed: 1, FOLD: 3, EPOCH: 14, train_loss: 0.007984, valid_loss: 0.008837, best_loss: 0.008837, best_loss_epoch: 14
    seed: 1, FOLD: 4, EPOCH: 0, train_loss: 0.647578, valid_loss: 0.217125, best_loss: 0.217125, best_loss_epoch: 0
    seed: 1, FOLD: 4, EPOCH: 10, train_loss: 0.008624, valid_loss: 0.008406, best_loss: 0.008406, best_loss_epoch: 10
    seed: 1, FOLD: 4, EPOCH: 14, train_loss: 0.008061, valid_loss: 0.008327, best_loss: 0.008324, best_loss_epoch: 13
    elapsed time: 210.8100562095642
    seed: 2, FOLD: 0, EPOCH: 0, train_loss: 0.649176, valid_loss: 0.232567, best_loss: 0.232567, best_loss_epoch: 0
    seed: 2, FOLD: 0, EPOCH: 10, train_loss: 0.008626, valid_loss: 0.008836, best_loss: 0.008836, best_loss_epoch: 10
    seed: 2, FOLD: 0, EPOCH: 14, train_loss: 0.008058, valid_loss: 0.008725, best_loss: 0.008725, best_loss_epoch: 14
    seed: 2, FOLD: 1, EPOCH: 0, train_loss: 0.648168, valid_loss: 0.217610, best_loss: 0.217610, best_loss_epoch: 0
    seed: 2, FOLD: 1, EPOCH: 10, train_loss: 0.008542, valid_loss: 0.008808, best_loss: 0.008808, best_loss_epoch: 10
    seed: 2, FOLD: 1, EPOCH: 14, train_loss: 0.008015, valid_loss: 0.008666, best_loss: 0.008666, best_loss_epoch: 14
    seed: 2, FOLD: 2, EPOCH: 0, train_loss: 0.648388, valid_loss: 0.233329, best_loss: 0.233329, best_loss_epoch: 0
    seed: 2, FOLD: 2, EPOCH: 10, train_loss: 0.008527, valid_loss: 0.008774, best_loss: 0.008774, best_loss_epoch: 10
    seed: 2, FOLD: 2, EPOCH: 14, train_loss: 0.008029, valid_loss: 0.008710, best_loss: 0.008710, best_loss_epoch: 14
    seed: 2, FOLD: 3, EPOCH: 0, train_loss: 0.647443, valid_loss: 0.200395, best_loss: 0.200395, best_loss_epoch: 0
    seed: 2, FOLD: 3, EPOCH: 10, train_loss: 0.008556, valid_loss: 0.008999, best_loss: 0.008999, best_loss_epoch: 10
    seed: 2, FOLD: 3, EPOCH: 14, train_loss: 0.008051, valid_loss: 0.008892, best_loss: 0.008892, best_loss_epoch: 14
    seed: 2, FOLD: 4, EPOCH: 0, train_loss: 0.646286, valid_loss: 0.254452, best_loss: 0.254452, best_loss_epoch: 0
    seed: 2, FOLD: 4, EPOCH: 10, train_loss: 0.008927, valid_loss: 0.008622, best_loss: 0.008622, best_loss_epoch: 10
    seed: 2, FOLD: 4, EPOCH: 14, train_loss: 0.008658, valid_loss: 0.008532, best_loss: 0.008532, best_loss_epoch: 14
    elapsed time: 312.43804264068604
    seed: 3, FOLD: 0, EPOCH: 0, train_loss: 0.648855, valid_loss: 0.230589, best_loss: 0.230589, best_loss_epoch: 0
    seed: 3, FOLD: 0, EPOCH: 10, train_loss: 0.008557, valid_loss: 0.008748, best_loss: 0.008748, best_loss_epoch: 10
    seed: 3, FOLD: 0, EPOCH: 14, train_loss: 0.008010, valid_loss: 0.008669, best_loss: 0.008669, best_loss_epoch: 14
    seed: 3, FOLD: 1, EPOCH: 0, train_loss: 0.647579, valid_loss: 0.218950, best_loss: 0.218950, best_loss_epoch: 0
    seed: 3, FOLD: 1, EPOCH: 10, train_loss: 0.008552, valid_loss: 0.008825, best_loss: 0.008825, best_loss_epoch: 10
    seed: 3, FOLD: 1, EPOCH: 14, train_loss: 0.008024, valid_loss: 0.008698, best_loss: 0.008698, best_loss_epoch: 14
    seed: 3, FOLD: 2, EPOCH: 0, train_loss: 0.646719, valid_loss: 0.256865, best_loss: 0.256865, best_loss_epoch: 0
    seed: 3, FOLD: 2, EPOCH: 10, train_loss: 0.008562, valid_loss: 0.008818, best_loss: 0.008818, best_loss_epoch: 10
    seed: 3, FOLD: 2, EPOCH: 14, train_loss: 0.008065, valid_loss: 0.008709, best_loss: 0.008706, best_loss_epoch: 13
    seed: 3, FOLD: 3, EPOCH: 0, train_loss: 0.647732, valid_loss: 0.208476, best_loss: 0.208476, best_loss_epoch: 0
    seed: 3, FOLD: 3, EPOCH: 10, train_loss: 0.008505, valid_loss: 0.008938, best_loss: 0.008938, best_loss_epoch: 10
    seed: 3, FOLD: 3, EPOCH: 14, train_loss: 0.008007, valid_loss: 0.008831, best_loss: 0.008831, best_loss_epoch: 14
    seed: 3, FOLD: 4, EPOCH: 0, train_loss: 0.647546, valid_loss: 0.215469, best_loss: 0.215469, best_loss_epoch: 0
    seed: 3, FOLD: 4, EPOCH: 10, train_loss: 0.008614, valid_loss: 0.008441, best_loss: 0.008441, best_loss_epoch: 10
    seed: 3, FOLD: 4, EPOCH: 14, train_loss: 0.008106, valid_loss: 0.008318, best_loss: 0.008318, best_loss_epoch: 14
    elapsed time: 414.71102643013
    seed: 4, FOLD: 0, EPOCH: 0, train_loss: 0.646602, valid_loss: 0.231037, best_loss: 0.231037, best_loss_epoch: 0
    seed: 4, FOLD: 0, EPOCH: 10, train_loss: 0.008581, valid_loss: 0.008774, best_loss: 0.008774, best_loss_epoch: 10
    seed: 4, FOLD: 0, EPOCH: 14, train_loss: 0.008133, valid_loss: 0.008718, best_loss: 0.008709, best_loss_epoch: 13
    seed: 4, FOLD: 1, EPOCH: 0, train_loss: 0.646445, valid_loss: 0.236030, best_loss: 0.236030, best_loss_epoch: 0
    seed: 4, FOLD: 1, EPOCH: 10, train_loss: 0.008568, valid_loss: 0.008826, best_loss: 0.008826, best_loss_epoch: 10
    seed: 4, FOLD: 1, EPOCH: 14, train_loss: 0.008090, valid_loss: 0.008700, best_loss: 0.008700, best_loss_epoch: 14
    seed: 4, FOLD: 2, EPOCH: 0, train_loss: 0.647786, valid_loss: 0.217754, best_loss: 0.217754, best_loss_epoch: 0
    seed: 4, FOLD: 2, EPOCH: 10, train_loss: 0.008748, valid_loss: 0.008912, best_loss: 0.008912, best_loss_epoch: 10
    seed: 4, FOLD: 2, EPOCH: 14, train_loss: 0.008306, valid_loss: 0.008852, best_loss: 0.008852, best_loss_epoch: 14
    seed: 4, FOLD: 3, EPOCH: 0, train_loss: 0.645685, valid_loss: 0.222522, best_loss: 0.222522, best_loss_epoch: 0
    seed: 4, FOLD: 3, EPOCH: 10, train_loss: 0.008514, valid_loss: 0.008968, best_loss: 0.008968, best_loss_epoch: 10
    seed: 4, FOLD: 3, EPOCH: 14, train_loss: 0.007985, valid_loss: 0.008832, best_loss: 0.008832, best_loss_epoch: 14
    seed: 4, FOLD: 4, EPOCH: 0, train_loss: 0.646384, valid_loss: 0.214957, best_loss: 0.214957, best_loss_epoch: 0
    seed: 4, FOLD: 4, EPOCH: 10, train_loss: 0.008657, valid_loss: 0.008435, best_loss: 0.008435, best_loss_epoch: 10
    seed: 4, FOLD: 4, EPOCH: 14, train_loss: 0.008179, valid_loss: 0.008329, best_loss: 0.008323, best_loss_epoch: 13
    elapsed time: 517.6590347290039
    (21948, 331)
    (3624, 331)



```python
train.to_pickle(f"{INT_DIR}/{NB}-train_nonscore_pred.pkl")
test.to_pickle(f"{INT_DIR}/{NB}-test_nonscore_pred.pkl")
```


```python
len(target_cols)
```




    331




```python
train[target_cols] = np.maximum(PMIN, np.minimum(PMAX, train[target_cols]))
valid_results = train_targets_nonscored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

y_true = train_targets_nonscored[target_cols].values
y_true = y_true > 0.5
y_pred = valid_results[target_cols].values

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]
    
print("CV log_loss: ", score)
```

    CV log_loss:  0.004833325424314168


CV log_loss:  0.014761779358699672
CV log_loss:  0.014519859174255039
CV log_loss:  0.014525173864593479
CV log_loss:  0.014354930596928602 # 3 umap features
CV log_loss:  0.014353604854355429 # more umap features
CV log_loss:  0.01436484670778641 # more hidden nodes


```python

EPOCHS = 25
# NFOLDS = 5

```


```python
# sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
# sub.to_csv('submission.csv', index=False)
```


```python
nonscored_target = [c for c in train[train_targets_nonscored.columns] if c != "sig_id"]
```


```python
nonscored_target
```




    ['abc_transporter_expression_enhancer',
     'abl_inhibitor',
     'ace_inhibitor',
     'acetylcholine_release_enhancer',
     'adenosine_kinase_inhibitor',
     'adenylyl_cyclase_inhibitor',
     'age_inhibitor',
     'alcohol_dehydrogenase_inhibitor',
     'aldehyde_dehydrogenase_activator',
     'aldose_reductase_inhibitor',
     'ampk_inhibitor',
     'androgen_biosynthesis_inhibitor',
     'angiotensin_receptor_agonist',
     'antacid',
     'anthelmintic',
     'antipruritic',
     'antirheumatic_drug',
     'antiseptic',
     'antispasmodic',
     'antithyroid_agent',
     'antitussive',
     'anxiolytic',
     'ap_inhibitor',
     'apoptosis_inhibitor',
     'arf_inhibitor',
     'aryl_hydrocarbon_receptor_agonist',
     'aryl_hydrocarbon_receptor_antagonist',
     'aspartic_protease_inhibitor',
     'atherogenesis_inhibitor',
     'atherosclerosis_formation_inhibitor',
     'atp-sensitive_potassium_channel_agonist',
     'atp-sensitive_potassium_channel_inhibitor',
     'atp_channel_blocker',
     'atp_citrase_lyase_inhibitor',
     'autophagy_inducer',
     'axl_kinase_inhibitor',
     'bacterial_atpase_inhibitor',
     'bacterial_permeability_inducer',
     'bacterial_protein_synthesis_inhibitor',
     'benzodiazepine_receptor_antagonist',
     'beta_catenin_inhibitor',
     'beta_lactamase_inhibitor',
     'beta_secretase_inhibitor',
     'big1_inhibitor',
     'bile_acid',
     'bone_resorption_inhibitor',
     'botulin_neurotoxin_inhibitor',
     'bradykinin_receptor_antagonist',
     'breast_cancer_resistance_protein_inhibitor',
     'bronchodilator',
     'calcitonin_antagonist',
     'calcium_channel_activator',
     'calmodulin_inhibitor',
     'camp_stimulant',
     'capillary_stabilizing_agent',
     'car_antagonist',
     'carcinogen',
     'carnitine_palmitoyltransferase_inhibitor',
     'caspase_inhibitor',
     'cathepsin_inhibitor',
     'cc_chemokine_receptor_agonist',
     'cdc_inhibitor',
     'cdk_expression_enhancer',
     'cell_cycle_inhibitor',
     'ceramidase_inhibitor',
     'cftr_channel_agonist',
     'chitin_inhibitor',
     'chloride_channel_activator',
     'choleretic_agent',
     'cholinergic_receptor_agonist',
     'cholinesterase_inhibitor',
     'clk_inhibitor',
     'coenzyme_a_precursor',
     'contraceptive_agent',
     'contrast_agent',
     'corticosteroid_antagonist',
     'cyclin_d_inhibitor',
     'cytidine_deaminase_inhibitor',
     'cytokine_production_inhibitor',
     'dehydrogenase_inhibitor',
     'diacylglycerol_kinase_inhibitor',
     'diacylglycerol_o_acyltransferase_inhibitor',
     'differentiation_inducer',
     'dihydroorotate_dehydrogenase_inhibitor',
     'dihydropteroate_synthase_inhibitor',
     'dihydropyrimidine_dehydrogenase_inhibitor',
     'dna_dependent_protein_kinase_inhibitor',
     'dna_methyltransferase_inhibitor',
     'dna_polymerase_inhibitor',
     'dna_synthesis_inhibitor',
     'dopamine_release_enhancer',
     'dot1l_inhibitor',
     'dynamin_inhibitor',
     'dyrk_inhibitor',
     'endothelin_receptor_antagonist',
     'enkephalinase_inhibitor',
     'ephrin_inhibitor',
     'epoxide_hydolase_inhibitor',
     'eukaryotic_translation_initiation_factor_inhibitor',
     'exportin_antagonist',
     'fabi_inhibitor',
     'farnesyl_pyrophosphate_synthase_inhibitor',
     'fatty_acid_receptor_antagonist',
     'fatty_acid_synthase_inhibitor',
     'folate_receptor_ligand',
     'fungal_ergosterol_inhibitor',
     'fungal_lanosterol_demethylase_inhibitor',
     'fxr_agonist',
     'g_protein-coupled_receptor_agonist',
     'g_protein-coupled_receptor_antagonist',
     'g_protein_signaling_inhibitor',
     'gaba_gated_chloride_channel_blocker',
     'gaba_receptor_modulator',
     'gaba_uptake_inhibitor',
     'gap_junction_modulator',
     'gastrin_inhibitor',
     'gat_inhibitor',
     'gli_antagonist',
     'glp_receptor_agonist',
     'glucagon_receptor_antagonist',
     'glucocorticoid_receptor_antagonist',
     'gluconeogenesis_inhibitor',
     'glucose_dependent_insulinotropic_receptor_agonist',
     'glucosidase_inhibitor',
     'glutamate_receptor_modulator',
     'glutathione_reductase_(nadph)_activators',
     'glycine_receptor_antagonist',
     'glycine_transporter_inhibitor',
     'glycogen_phosphorylase_inhibitor',
     'glycosylation_inhibitor',
     'gonadotropin_receptor_antagonist',
     'growth_factor_receptor_inhibitor',
     'gtpase_inhibitor',
     'guanylate_cyclase_activator',
     'guanylyl_cyclase_activator',
     'haemostatic_agent',
     'hcn_channel_antagonist',
     'hedgehog_pathway_inhibitor',
     'heme_oxygenase_activators',
     'hemoglobin_antagonist',
     'hexokinase_inhibitor',
     'hgf_receptor_inhibitor',
     'hif_inhibitor',
     'histamine_release_inhibitor',
     'histone_acetyltransferase_inhibitor',
     'histone_demethylase_inhibitor',
     'hiv_integrase_inhibitor',
     'hiv_protease_inhibitor',
     'hydantoin_antiepileptic',
     'hydroxycarboxylic_acid_receptor_agonist',
     'icam1_antagonist',
     'icam1_inhibitor',
     'id1_expression_inhibitor',
     'imidazoline_ligand',
     'immunostimulant',
     'indoleamine_2,3-dioxygenase_inhibitor',
     'inosine_monophosphate_dehydrogenase_inhibitor',
     'inositol_monophosphatase_inhibitor',
     'interferon_inducer',
     'interleukin_inhibitor',
     'interleukin_receptor_agonist',
     'ion_channel_antagonist',
     'ip1_prostacyclin_receptor_agonist',
     'isocitrate_dehydrogenase_inhibitor',
     'jnk_inhibitor',
     'kainate_receptor_antagonist',
     'katp_activator',
     'keap1_ligand',
     'kinesin_inhibitor',
     'lactamase_inhibitor',
     'lactate_dehydrogenase_inhibitor',
     'lanosterol_demethylase_inhibitor',
     'leucyl-trna_synthetase_inhibitor',
     'leukocyte_elastase_inhibitor',
     'lim_inhibitor',
     'lipase_clearing_factor_inhibitor',
     'lipid_peroxidase_inhibitor',
     'lipoprotein_lipase_activator',
     'lrkk2_inhibitor',
     'lymphocyte_inhibitor',
     'lysophosphatidic_acid_receptor_antagonist',
     'macrophage_inhibitor',
     'macrophage_migration_inhibiting_factor_inhibitor',
     'map_k',
     'map_kinase_inhibitor',
     'matrix_metalloprotease_inhibitor',
     'mcl1_inhibitor',
     'melanin_inhibitor',
     'melatonin_receptor_agonist',
     'membrane_permeability_inhibitor',
     'mer_tyrosine_kinase_inhibitor',
     'met_inhibitor',
     'metalloproteinase_inhibitor',
     'mineralocorticoid_receptor_agonist',
     'mitochondrial_inhibitor',
     'mitochondrial_na+_ca2+_exchanger_antagonist',
     'monocarboxylate_transporter_inhibitor',
     'motilin_receptor_agonist',
     'mrp_inhibitor',
     'mth1_inhibitor',
     'mucus_protecting_agent',
     'muscle_relaxant',
     'na_k-atpase_inhibitor',
     'nadph_inhibitor',
     'nampt_inhibitor',
     'neprilysin_inhibitor',
     'neural_stem_cell_inducer',
     'neuraminidase_inhibitor',
     'neurokinin_receptor_antagonist',
     'neurotensin_receptor_agonist',
     'neurotensin_receptor_antagonist',
     'neurotransmitter',
     'neurotrophic_agent',
     'nfkb_activator',
     'niemann-pick_c1-like_1_protein_antagonist',
     'nitric_oxide_scavenger',
     'nociceptin_orphanin_fq_(nop)_receptor_antagonist',
     'non-nucleoside_reverse_transcriptase_inhibitor',
     'nootropic_agent',
     'nop_receptor_agonist',
     'norepinephrine_inhibitor',
     'notch_signaling_inhibitor',
     'nucleoside_reverse_transcriptase_inhibitor',
     'oct_activator',
     'omega_3_fatty_acid_stimulant',
     'osteoclast_inhibitor',
     'oxidosqualene_cyclase_inhibitor',
     'oxytocin_receptor_agonist',
     'oxytocin_receptor_antagonist',
     'p21_activated_kinase_inhibitor',
     'p53_activator',
     'p53_inhibitor',
     'paba_antagonist',
     'pdk1_inhibitor',
     'penicillin_binding_protein_inhibitor',
     'peptidase_inhibitor',
     'perk_inhibitor',
     'phosphofructokinase_inhibitor',
     'phospholipase_activator',
     'pim_inhibitor',
     'pka_inhibitor',
     'plasminogen_activator_inhibitor',
     'platelet_activating_factor_receptor_antagonist',
     'platelet_aggregation_inhibitor',
     'plk_inhibitor',
     'porcupine_inhibitor',
     'potassium_channel_agonist',
     'potassium_channel_blocker',
     'prmt_inhibitor',
     'progestogen_hormone',
     'prolactin_inhibitor',
     'prostacyclin_analog',
     'prostanoid_receptor_agonist',
     'protease_inhibitor',
     'protein_kinase_activator',
     'protein_synthesis_stimulant',
     'psychoactive_drug',
     'purine_antagonist',
     'purinergic_receptor_antagonist',
     'quorum_sensing_signaling_modulator',
     'rad51_inhibitor',
     'receptor_tyrosine_protein_kinase_inhibitor',
     'reducing_agent',
     'ret_inhibitor',
     'ret_tyrosine_kinase_inhibitor',
     'reverse_transcriptase_inhibitor',
     'ribosomal_protein_inhibitor',
     'rna_synthesis_inhibitor',
     'ror_inverse_agonist',
     'rsv_fusion_inhibitor',
     'sars_coronavirus_3c-like_protease_inhibitor',
     'sedative',
     'selective_estrogen_receptor_modulator_(serm)',
     'selective_serotonin_reuptake_inhibitor_(ssri)',
     'serine_protease_inhibitor',
     'serine_threonine_kinase_inhibitor',
     'serotonin_release_inhibitor',
     'sirt_activator',
     'sirt_inhibitor',
     'smoothened_receptor_agonist',
     'sodium_calcium_exchange_inhibitor',
     'sodium_channel_activator',
     'sodium_channel_blocker',
     'somatostatin_receptor_agonist',
     'sphingosine_1_phosphate_receptor_agonist',
     'sphingosine_kinase_inhibitor',
     'srebp_inhibitor',
     'stat_inhibitor',
     'steroid_sulfatase_inhibitor',
     'steroidal_progestin',
     'sterol_demethylase_inhibitor',
     'steryl_sulfatase_inhibitor',
     'structural_glycoprotein_antagonist',
     'succinimide_antiepileptic',
     't_cell_inhibitor',
     'tankyrase_inhibitor',
     'telomerase_inhibitor',
     'testosterone_receptor_antagonist',
     'thiazide_diuretic',
     'thrombopoietin_receptor_agonist',
     'thromboxane_receptor_antagonist',
     'thromboxane_synthase_inhibitor',
     'thyroid_hormone_inhibitor',
     'thyroid_hormone_stimulant',
     'thyrotropin_releasing_hormone_receptor_agonist',
     'tissue_transglutaminase_inhibitor',
     'topical_sunscreen_agent',
     'trace_amine_associated_receptor_agonist',
     'triacylglycerol_lipase_inhibitor',
     'tricyclic_antidepressant',
     'tryptophan_hydroxylase_inhibitor',
     'tyrosinase_inhibitor',
     'tyrosine_hydroxylase_inhibitor',
     'tyrosine_phosphatase_inhibitor',
     'ubiquitin-conjugating_enzyme_inhibitor',
     'urease_inhibitor',
     'uricase_inhibitor',
     'uricosuric',
     'urotensin_receptor_antagonist',
     'vasoconstrictor',
     'vasodilator',
     'vasopressin_receptor_agonist',
     'vasopressin_receptor_antagonist',
     've-cadherin_antagonist',
     'vesicular_monoamine_transporter_inhibitor',
     'vitamin_k_antagonist',
     'voltage-gated_potassium_channel_activator',
     'voltage-gated_sodium_channel_blocker',
     'wdr5_mll_interaction_inhibitor',
     'xanthine_oxidase_inhibitor',
     'xiap_inhibitor']




```python
train = pd.read_pickle(f"{INT_DIR}/{NB}-train_nonscore_pred.pkl")
test = pd.read_pickle(f"{INT_DIR}/{NB}-test_nonscore_pred.pkl")
```


```python
# use nonscored target in the given file as feature
# if comment out below, use predicted nonscored target
# train = train.drop(nonscored_target, axis=1)
# train = train.merge(train_targets_nonscored, on="sig_id")
# train = train_features.merge(train_targets_scored, on='sig_id')
train = train.merge(train_targets_scored, on='sig_id')
# train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
# test = test[test['cp_type']!='ctl_vehicle'].reset_index(drop=True)

# target = train[train_targets_scored.columns]
target = train[train_targets_scored.columns]
```


```python
# from sklearn.preprocessing import QuantileTransformer

for col in (nonscored_target):

    vec_len = len(train[col].values)
    vec_len_test = len(test[col].values)
    raw_vec = train[col].values.reshape(vec_len, 1)
    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        transformer.fit(raw_vec)
        pd.to_pickle(transformer, f"{MODEL_DIR}/{NB}_{col}_quantile_nonscored.pkl")
    else:
        transformer = pd.read_pickle(f"{MODEL_DIR}/{NB}_{col}_quantile_nonscored.pkl")

    train[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test[col] = transformer.transform(test[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]
```


```python
target_cols = target.drop('sig_id', axis=1).columns.values.tolist()
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>g-0</th>
      <th>g-1</th>
      <th>g-2</th>
      <th>g-3</th>
      <th>g-4</th>
      <th>g-5</th>
      <th>g-6</th>
      <th>...</th>
      <th>tropomyosin_receptor_kinase_inhibitor</th>
      <th>trpv_agonist</th>
      <th>trpv_antagonist</th>
      <th>tubulin_inhibitor</th>
      <th>tyrosine_kinase_inhibitor</th>
      <th>ubiquitin_specific_protease_inhibitor</th>
      <th>vegfr_inhibitor</th>
      <th>vitamin_b</th>
      <th>vitamin_d_receptor_agonist</th>
      <th>wnt_inhibitor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>24</td>
      <td>D1</td>
      <td>1.146806</td>
      <td>0.902075</td>
      <td>-0.418339</td>
      <td>-0.961202</td>
      <td>-0.254770</td>
      <td>-1.021300</td>
      <td>-1.369236</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>72</td>
      <td>D1</td>
      <td>0.128824</td>
      <td>0.676862</td>
      <td>0.274345</td>
      <td>0.090495</td>
      <td>1.208863</td>
      <td>0.688965</td>
      <td>0.316734</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_000a6266a</td>
      <td>48</td>
      <td>D1</td>
      <td>0.790372</td>
      <td>0.939951</td>
      <td>1.428097</td>
      <td>-0.121817</td>
      <td>-0.002067</td>
      <td>1.495091</td>
      <td>0.238763</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_0015fd391</td>
      <td>48</td>
      <td>D1</td>
      <td>-0.729866</td>
      <td>-0.277163</td>
      <td>-0.441200</td>
      <td>0.766612</td>
      <td>2.347817</td>
      <td>-0.862761</td>
      <td>-2.308829</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_001626bd3</td>
      <td>72</td>
      <td>D2</td>
      <td>-0.444558</td>
      <td>-0.481202</td>
      <td>0.974729</td>
      <td>0.977467</td>
      <td>1.468304</td>
      <td>-0.874772</td>
      <td>-0.372682</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21943</th>
      <td>id_fff8c2444</td>
      <td>72</td>
      <td>D1</td>
      <td>0.247623</td>
      <td>-1.231184</td>
      <td>0.221572</td>
      <td>-0.354096</td>
      <td>-0.332073</td>
      <td>0.570635</td>
      <td>-0.150125</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21944</th>
      <td>id_fffb1ceed</td>
      <td>24</td>
      <td>D2</td>
      <td>0.217613</td>
      <td>-0.027031</td>
      <td>-0.237430</td>
      <td>-0.787215</td>
      <td>-0.677817</td>
      <td>0.919474</td>
      <td>0.742866</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21945</th>
      <td>id_fffb70c0c</td>
      <td>24</td>
      <td>D2</td>
      <td>-1.914666</td>
      <td>0.581880</td>
      <td>-0.588706</td>
      <td>1.303439</td>
      <td>-1.009079</td>
      <td>0.852202</td>
      <td>-0.302814</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21946</th>
      <td>id_fffcb9e7c</td>
      <td>24</td>
      <td>D1</td>
      <td>0.826302</td>
      <td>0.411235</td>
      <td>0.433297</td>
      <td>0.307575</td>
      <td>1.075324</td>
      <td>-0.024425</td>
      <td>0.051483</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21947</th>
      <td>id_ffffdd77b</td>
      <td>72</td>
      <td>D1</td>
      <td>-1.245739</td>
      <td>1.567230</td>
      <td>-0.269829</td>
      <td>1.092958</td>
      <td>-0.515819</td>
      <td>-2.091765</td>
      <td>-1.627645</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>21948 rows × 1622 columns</p>
</div>




```python
folds = train.copy()

mskf = MultilabelStratifiedKFold(n_splits=NFOLDS)

for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
    folds.loc[v_idx, 'kfold'] = int(f)

folds['kfold'] = folds['kfold'].astype(int)
folds
```

    /opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py:70: FutureWarning: Pass shuffle=False, random_state=None as keyword args. From version 0.25 passing these as positional arguments will result in an error
      FutureWarning)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>cp_time</th>
      <th>cp_dose</th>
      <th>g-0</th>
      <th>g-1</th>
      <th>g-2</th>
      <th>g-3</th>
      <th>g-4</th>
      <th>g-5</th>
      <th>g-6</th>
      <th>...</th>
      <th>trpv_agonist</th>
      <th>trpv_antagonist</th>
      <th>tubulin_inhibitor</th>
      <th>tyrosine_kinase_inhibitor</th>
      <th>ubiquitin_specific_protease_inhibitor</th>
      <th>vegfr_inhibitor</th>
      <th>vitamin_b</th>
      <th>vitamin_d_receptor_agonist</th>
      <th>wnt_inhibitor</th>
      <th>kfold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>24</td>
      <td>D1</td>
      <td>1.146806</td>
      <td>0.902075</td>
      <td>-0.418339</td>
      <td>-0.961202</td>
      <td>-0.254770</td>
      <td>-1.021300</td>
      <td>-1.369236</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>72</td>
      <td>D1</td>
      <td>0.128824</td>
      <td>0.676862</td>
      <td>0.274345</td>
      <td>0.090495</td>
      <td>1.208863</td>
      <td>0.688965</td>
      <td>0.316734</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_000a6266a</td>
      <td>48</td>
      <td>D1</td>
      <td>0.790372</td>
      <td>0.939951</td>
      <td>1.428097</td>
      <td>-0.121817</td>
      <td>-0.002067</td>
      <td>1.495091</td>
      <td>0.238763</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_0015fd391</td>
      <td>48</td>
      <td>D1</td>
      <td>-0.729866</td>
      <td>-0.277163</td>
      <td>-0.441200</td>
      <td>0.766612</td>
      <td>2.347817</td>
      <td>-0.862761</td>
      <td>-2.308829</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_001626bd3</td>
      <td>72</td>
      <td>D2</td>
      <td>-0.444558</td>
      <td>-0.481202</td>
      <td>0.974729</td>
      <td>0.977467</td>
      <td>1.468304</td>
      <td>-0.874772</td>
      <td>-0.372682</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21943</th>
      <td>id_fff8c2444</td>
      <td>72</td>
      <td>D1</td>
      <td>0.247623</td>
      <td>-1.231184</td>
      <td>0.221572</td>
      <td>-0.354096</td>
      <td>-0.332073</td>
      <td>0.570635</td>
      <td>-0.150125</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>21944</th>
      <td>id_fffb1ceed</td>
      <td>24</td>
      <td>D2</td>
      <td>0.217613</td>
      <td>-0.027031</td>
      <td>-0.237430</td>
      <td>-0.787215</td>
      <td>-0.677817</td>
      <td>0.919474</td>
      <td>0.742866</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21945</th>
      <td>id_fffb70c0c</td>
      <td>24</td>
      <td>D2</td>
      <td>-1.914666</td>
      <td>0.581880</td>
      <td>-0.588706</td>
      <td>1.303439</td>
      <td>-1.009079</td>
      <td>0.852202</td>
      <td>-0.302814</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>21946</th>
      <td>id_fffcb9e7c</td>
      <td>24</td>
      <td>D1</td>
      <td>0.826302</td>
      <td>0.411235</td>
      <td>0.433297</td>
      <td>0.307575</td>
      <td>1.075324</td>
      <td>-0.024425</td>
      <td>0.051483</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21947</th>
      <td>id_ffffdd77b</td>
      <td>72</td>
      <td>D1</td>
      <td>-1.245739</td>
      <td>1.567230</td>
      <td>-0.269829</td>
      <td>1.092958</td>
      <td>-0.515819</td>
      <td>-2.091765</td>
      <td>-1.627645</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>21948 rows × 1623 columns</p>
</div>




```python
print(train.shape)
print(folds.shape)
print(test.shape)
print(target.shape)
print(sample_submission.shape)
```

    (21948, 1622)
    (21948, 1623)
    (3624, 1416)
    (21948, 207)
    (3982, 207)



```python
def process_data(data):
    
    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
#     data.loc[:, 'cp_time'] = data.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})
#     data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

# --------------------- Normalize ---------------------
#     for col in GENES:
#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))
    
#     for col in CELLS:
#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))
    
#--------------------- Removing Skewness ---------------------
#     for col in GENES + CELLS:
#         if(abs(data[col].skew()) > 0.75):
            
#             if(data[col].skew() < 0): # neg-skewness
#                 data[col] = data[col].max() - data[col] + 1
#                 data[col] = np.sqrt(data[col])
            
#             else:
#                 data[col] = np.sqrt(data[col])
    
    return data
```


```python
feature_cols = [c for c in process_data(folds).columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]
len(feature_cols)
```




    1418




```python
num_features=len(feature_cols)
num_targets=len(target_cols)
hidden_size=2048
# hidden_size=4096
# hidden_size=9192
```


```python
def run_training(fold, seed):
    
    seed_everything(seed)
    
    train = process_data(folds)
    test_ = process_data(test)
    
    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index
    
    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)
    
    x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values
    
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.3, div_factor=1000, 
#                                               max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.2, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    loss_fn = nn.BCEWithLogitsLoss()
    
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
    
    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    best_loss = np.inf
    best_loss_epoch = -1
    
    if IS_TRAIN:
        for epoch in range(EPOCHS):

            train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)

            if valid_loss < best_loss:            
                best_loss = valid_loss
                best_loss_epoch = epoch
                oof[val_idx] = valid_preds
                torch.save(model.state_dict(), f"{MODEL_DIR}/{NB}-scored-SEED{seed}-FOLD{fold}_.pth")

            elif(EARLY_STOP == True):
                early_step += 1
                if (early_step >= early_stopping_steps):
                    break

            if epoch % 10 == 0 or epoch == EPOCHS-1:
                print(f"seed: {seed}, FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}, best_loss: {best_loss:.6f}, best_loss_epoch: {best_loss_epoch}")            
   
    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.load_state_dict(torch.load(f"{MODEL_DIR}/{NB}-scored-SEED{seed}-FOLD{fold}_.pth"))
    model.to(DEVICE)
    
    if not IS_TRAIN:
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
        oof[val_idx] = valid_preds    
    
    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)
    
    return oof, predictions
```


```python
def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))
    
    for fold in range(NFOLDS):
        oof_, pred_ = run_training(fold, seed)
        
        predictions += pred_ / NFOLDS
        oof += oof_
        
    return oof, predictions
```


```python
SEED = range(NSEEDS)  #[0, 1, 2, 3 ,4]#, 5, 6, 7, 8, 9, 10]
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

time_start = time.time()

for seed in SEED:
    
    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)
    print(f"elapsed time: {time.time() - time_start}")

train[target_cols] = oof
test[target_cols] = predictions
```

    seed: 0, FOLD: 0, EPOCH: 0, train_loss: 0.713415, valid_loss: 0.616348, best_loss: 0.616348, best_loss_epoch: 0
    seed: 0, FOLD: 0, EPOCH: 10, train_loss: 0.016176, valid_loss: 0.016713, best_loss: 0.016713, best_loss_epoch: 10
    seed: 0, FOLD: 0, EPOCH: 20, train_loss: 0.013328, valid_loss: 0.015798, best_loss: 0.015798, best_loss_epoch: 20
    seed: 0, FOLD: 0, EPOCH: 24, train_loss: 0.011210, valid_loss: 0.015821, best_loss: 0.015798, best_loss_epoch: 20
    seed: 0, FOLD: 1, EPOCH: 0, train_loss: 0.713643, valid_loss: 0.610121, best_loss: 0.610121, best_loss_epoch: 0
    seed: 0, FOLD: 1, EPOCH: 10, train_loss: 0.016095, valid_loss: 0.016652, best_loss: 0.016652, best_loss_epoch: 10
    seed: 0, FOLD: 1, EPOCH: 20, train_loss: 0.013382, valid_loss: 0.015891, best_loss: 0.015891, best_loss_epoch: 20
    seed: 0, FOLD: 1, EPOCH: 24, train_loss: 0.011330, valid_loss: 0.015828, best_loss: 0.015820, best_loss_epoch: 23
    seed: 0, FOLD: 2, EPOCH: 0, train_loss: 0.713401, valid_loss: 0.622936, best_loss: 0.622936, best_loss_epoch: 0
    seed: 0, FOLD: 2, EPOCH: 10, train_loss: 0.016291, valid_loss: 0.016424, best_loss: 0.016424, best_loss_epoch: 10
    seed: 0, FOLD: 2, EPOCH: 20, train_loss: 0.013424, valid_loss: 0.015751, best_loss: 0.015751, best_loss_epoch: 20
    seed: 0, FOLD: 2, EPOCH: 24, train_loss: 0.011468, valid_loss: 0.015680, best_loss: 0.015679, best_loss_epoch: 23
    seed: 0, FOLD: 3, EPOCH: 0, train_loss: 0.713308, valid_loss: 0.618372, best_loss: 0.618372, best_loss_epoch: 0
    seed: 0, FOLD: 3, EPOCH: 10, train_loss: 0.016180, valid_loss: 0.016714, best_loss: 0.016480, best_loss_epoch: 9
    seed: 0, FOLD: 3, EPOCH: 20, train_loss: 0.013380, valid_loss: 0.015770, best_loss: 0.015770, best_loss_epoch: 20
    seed: 0, FOLD: 3, EPOCH: 24, train_loss: 0.011403, valid_loss: 0.015719, best_loss: 0.015719, best_loss_epoch: 24
    seed: 0, FOLD: 4, EPOCH: 0, train_loss: 0.712820, valid_loss: 0.614307, best_loss: 0.614307, best_loss_epoch: 0
    seed: 0, FOLD: 4, EPOCH: 10, train_loss: 0.016228, valid_loss: 0.016824, best_loss: 0.016718, best_loss_epoch: 9
    seed: 0, FOLD: 4, EPOCH: 20, train_loss: 0.013272, valid_loss: 0.015823, best_loss: 0.015823, best_loss_epoch: 20
    seed: 0, FOLD: 4, EPOCH: 24, train_loss: 0.011206, valid_loss: 0.015813, best_loss: 0.015776, best_loss_epoch: 21
    elapsed time: 180.47887063026428
    seed: 1, FOLD: 0, EPOCH: 0, train_loss: 0.713063, valid_loss: 0.614382, best_loss: 0.614382, best_loss_epoch: 0
    seed: 1, FOLD: 0, EPOCH: 10, train_loss: 0.016173, valid_loss: 0.016646, best_loss: 0.016646, best_loss_epoch: 10
    seed: 1, FOLD: 0, EPOCH: 20, train_loss: 0.013190, valid_loss: 0.015784, best_loss: 0.015784, best_loss_epoch: 20
    seed: 1, FOLD: 0, EPOCH: 24, train_loss: 0.011143, valid_loss: 0.015811, best_loss: 0.015784, best_loss_epoch: 20
    seed: 1, FOLD: 1, EPOCH: 0, train_loss: 0.712554, valid_loss: 0.607957, best_loss: 0.607957, best_loss_epoch: 0
    seed: 1, FOLD: 1, EPOCH: 10, train_loss: 0.016614, valid_loss: 0.016866, best_loss: 0.016866, best_loss_epoch: 10
    seed: 1, FOLD: 1, EPOCH: 20, train_loss: 0.013065, valid_loss: 0.015995, best_loss: 0.015982, best_loss_epoch: 19
    seed: 1, FOLD: 1, EPOCH: 24, train_loss: 0.010661, valid_loss: 0.016020, best_loss: 0.015982, best_loss_epoch: 19
    seed: 1, FOLD: 2, EPOCH: 0, train_loss: 0.712389, valid_loss: 0.594676, best_loss: 0.594676, best_loss_epoch: 0
    seed: 1, FOLD: 2, EPOCH: 10, train_loss: 0.016120, valid_loss: 0.016331, best_loss: 0.016331, best_loss_epoch: 10
    seed: 1, FOLD: 2, EPOCH: 20, train_loss: 0.013153, valid_loss: 0.015716, best_loss: 0.015716, best_loss_epoch: 20
    seed: 1, FOLD: 2, EPOCH: 24, train_loss: 0.010990, valid_loss: 0.015693, best_loss: 0.015646, best_loss_epoch: 21
    seed: 1, FOLD: 3, EPOCH: 0, train_loss: 0.712980, valid_loss: 0.608345, best_loss: 0.608345, best_loss_epoch: 0
    seed: 1, FOLD: 3, EPOCH: 10, train_loss: 0.016235, valid_loss: 0.016600, best_loss: 0.016600, best_loss_epoch: 10
    seed: 1, FOLD: 3, EPOCH: 20, train_loss: 0.013323, valid_loss: 0.015728, best_loss: 0.015728, best_loss_epoch: 20
    seed: 1, FOLD: 3, EPOCH: 24, train_loss: 0.011254, valid_loss: 0.015675, best_loss: 0.015619, best_loss_epoch: 21
    seed: 1, FOLD: 4, EPOCH: 0, train_loss: 0.712315, valid_loss: 0.589346, best_loss: 0.589346, best_loss_epoch: 0
    seed: 1, FOLD: 4, EPOCH: 10, train_loss: 0.016189, valid_loss: 0.016723, best_loss: 0.016602, best_loss_epoch: 9
    seed: 1, FOLD: 4, EPOCH: 20, train_loss: 0.013198, valid_loss: 0.015783, best_loss: 0.015783, best_loss_epoch: 20
    seed: 1, FOLD: 4, EPOCH: 24, train_loss: 0.011087, valid_loss: 0.015759, best_loss: 0.015759, best_loss_epoch: 24
    elapsed time: 357.22492384910583
    seed: 2, FOLD: 0, EPOCH: 0, train_loss: 0.713505, valid_loss: 0.595237, best_loss: 0.595237, best_loss_epoch: 0
    seed: 2, FOLD: 0, EPOCH: 10, train_loss: 0.016050, valid_loss: 0.016573, best_loss: 0.016573, best_loss_epoch: 10
    seed: 2, FOLD: 0, EPOCH: 20, train_loss: 0.012970, valid_loss: 0.015911, best_loss: 0.015911, best_loss_epoch: 20
    seed: 2, FOLD: 0, EPOCH: 24, train_loss: 0.010749, valid_loss: 0.015921, best_loss: 0.015875, best_loss_epoch: 22
    seed: 2, FOLD: 1, EPOCH: 0, train_loss: 0.713470, valid_loss: 0.598378, best_loss: 0.598378, best_loss_epoch: 0
    seed: 2, FOLD: 1, EPOCH: 10, train_loss: 0.016208, valid_loss: 0.016769, best_loss: 0.016768, best_loss_epoch: 8
    seed: 2, FOLD: 1, EPOCH: 20, train_loss: 0.013219, valid_loss: 0.015840, best_loss: 0.015840, best_loss_epoch: 20
    seed: 2, FOLD: 1, EPOCH: 24, train_loss: 0.011218, valid_loss: 0.015798, best_loss: 0.015779, best_loss_epoch: 23
    seed: 2, FOLD: 2, EPOCH: 0, train_loss: 0.713456, valid_loss: 0.614738, best_loss: 0.614738, best_loss_epoch: 0
    seed: 2, FOLD: 2, EPOCH: 10, train_loss: 0.017095, valid_loss: 0.017032, best_loss: 0.017032, best_loss_epoch: 10
    seed: 2, FOLD: 2, EPOCH: 20, train_loss: 0.013982, valid_loss: 0.015700, best_loss: 0.015700, best_loss_epoch: 20
    seed: 2, FOLD: 2, EPOCH: 24, train_loss: 0.011963, valid_loss: 0.015719, best_loss: 0.015700, best_loss_epoch: 20
    seed: 2, FOLD: 3, EPOCH: 0, train_loss: 0.714079, valid_loss: 0.622078, best_loss: 0.622078, best_loss_epoch: 0
    seed: 2, FOLD: 3, EPOCH: 10, train_loss: 0.016156, valid_loss: 0.016553, best_loss: 0.016472, best_loss_epoch: 8
    seed: 2, FOLD: 3, EPOCH: 20, train_loss: 0.013189, valid_loss: 0.015749, best_loss: 0.015749, best_loss_epoch: 20
    seed: 2, FOLD: 3, EPOCH: 24, train_loss: 0.011088, valid_loss: 0.015700, best_loss: 0.015700, best_loss_epoch: 24
    seed: 2, FOLD: 4, EPOCH: 0, train_loss: 0.713886, valid_loss: 0.607125, best_loss: 0.607125, best_loss_epoch: 0
    seed: 2, FOLD: 4, EPOCH: 10, train_loss: 0.016202, valid_loss: 0.016594, best_loss: 0.016594, best_loss_epoch: 10
    seed: 2, FOLD: 4, EPOCH: 20, train_loss: 0.013299, valid_loss: 0.015816, best_loss: 0.015816, best_loss_epoch: 20
    seed: 2, FOLD: 4, EPOCH: 24, train_loss: 0.011156, valid_loss: 0.015785, best_loss: 0.015773, best_loss_epoch: 22
    elapsed time: 536.2732951641083
    seed: 3, FOLD: 0, EPOCH: 0, train_loss: 0.714044, valid_loss: 0.613843, best_loss: 0.613843, best_loss_epoch: 0
    seed: 3, FOLD: 0, EPOCH: 10, train_loss: 0.016112, valid_loss: 0.016612, best_loss: 0.016612, best_loss_epoch: 10
    seed: 3, FOLD: 0, EPOCH: 20, train_loss: 0.013213, valid_loss: 0.015841, best_loss: 0.015841, best_loss_epoch: 20
    seed: 3, FOLD: 0, EPOCH: 24, train_loss: 0.011154, valid_loss: 0.015813, best_loss: 0.015772, best_loss_epoch: 22
    seed: 3, FOLD: 1, EPOCH: 0, train_loss: 0.713834, valid_loss: 0.624007, best_loss: 0.624007, best_loss_epoch: 0
    seed: 3, FOLD: 1, EPOCH: 10, train_loss: 0.016170, valid_loss: 0.016546, best_loss: 0.016546, best_loss_epoch: 10
    seed: 3, FOLD: 1, EPOCH: 20, train_loss: 0.013206, valid_loss: 0.015940, best_loss: 0.015940, best_loss_epoch: 20
    seed: 3, FOLD: 1, EPOCH: 24, train_loss: 0.011146, valid_loss: 0.015857, best_loss: 0.015834, best_loss_epoch: 22
    seed: 3, FOLD: 2, EPOCH: 0, train_loss: 0.713676, valid_loss: 0.614863, best_loss: 0.614863, best_loss_epoch: 0
    seed: 3, FOLD: 2, EPOCH: 10, train_loss: 0.016175, valid_loss: 0.016537, best_loss: 0.016537, best_loss_epoch: 10
    seed: 3, FOLD: 2, EPOCH: 20, train_loss: 0.013334, valid_loss: 0.015849, best_loss: 0.015747, best_loss_epoch: 19
    seed: 3, FOLD: 2, EPOCH: 24, train_loss: 0.011280, valid_loss: 0.015714, best_loss: 0.015668, best_loss_epoch: 21
    seed: 3, FOLD: 3, EPOCH: 0, train_loss: 0.714098, valid_loss: 0.622541, best_loss: 0.622541, best_loss_epoch: 0
    seed: 3, FOLD: 3, EPOCH: 10, train_loss: 0.016199, valid_loss: 0.016566, best_loss: 0.016566, best_loss_epoch: 10
    seed: 3, FOLD: 3, EPOCH: 20, train_loss: 0.013246, valid_loss: 0.015746, best_loss: 0.015746, best_loss_epoch: 20
    seed: 3, FOLD: 3, EPOCH: 24, train_loss: 0.011249, valid_loss: 0.015683, best_loss: 0.015672, best_loss_epoch: 22
    seed: 3, FOLD: 4, EPOCH: 0, train_loss: 0.714221, valid_loss: 0.607972, best_loss: 0.607972, best_loss_epoch: 0
    seed: 3, FOLD: 4, EPOCH: 10, train_loss: 0.016195, valid_loss: 0.016694, best_loss: 0.016694, best_loss_epoch: 10
    seed: 3, FOLD: 4, EPOCH: 20, train_loss: 0.013127, valid_loss: 0.015843, best_loss: 0.015843, best_loss_epoch: 20
    seed: 3, FOLD: 4, EPOCH: 24, train_loss: 0.011060, valid_loss: 0.015758, best_loss: 0.015758, best_loss_epoch: 24
    elapsed time: 724.633293390274
    seed: 4, FOLD: 0, EPOCH: 0, train_loss: 0.713043, valid_loss: 0.589019, best_loss: 0.589019, best_loss_epoch: 0
    seed: 4, FOLD: 0, EPOCH: 10, train_loss: 0.016196, valid_loss: 0.016957, best_loss: 0.016661, best_loss_epoch: 9
    seed: 4, FOLD: 0, EPOCH: 20, train_loss: 0.013247, valid_loss: 0.015900, best_loss: 0.015876, best_loss_epoch: 17
    seed: 4, FOLD: 0, EPOCH: 24, train_loss: 0.011165, valid_loss: 0.015758, best_loss: 0.015758, best_loss_epoch: 24
    seed: 4, FOLD: 1, EPOCH: 0, train_loss: 0.712967, valid_loss: 0.591954, best_loss: 0.591954, best_loss_epoch: 0
    seed: 4, FOLD: 1, EPOCH: 10, train_loss: 0.016094, valid_loss: 0.016579, best_loss: 0.016579, best_loss_epoch: 10
    seed: 4, FOLD: 1, EPOCH: 20, train_loss: 0.013284, valid_loss: 0.015897, best_loss: 0.015897, best_loss_epoch: 20
    seed: 4, FOLD: 1, EPOCH: 24, train_loss: 0.011311, valid_loss: 0.015863, best_loss: 0.015863, best_loss_epoch: 24
    seed: 4, FOLD: 2, EPOCH: 0, train_loss: 0.712839, valid_loss: 0.591046, best_loss: 0.591046, best_loss_epoch: 0
    seed: 4, FOLD: 2, EPOCH: 10, train_loss: 0.016290, valid_loss: 0.016509, best_loss: 0.016509, best_loss_epoch: 10
    seed: 4, FOLD: 2, EPOCH: 20, train_loss: 0.013291, valid_loss: 0.015681, best_loss: 0.015681, best_loss_epoch: 20
    seed: 4, FOLD: 2, EPOCH: 24, train_loss: 0.011212, valid_loss: 0.015619, best_loss: 0.015597, best_loss_epoch: 22
    seed: 4, FOLD: 3, EPOCH: 0, train_loss: 0.712762, valid_loss: 0.605229, best_loss: 0.605229, best_loss_epoch: 0
    seed: 4, FOLD: 3, EPOCH: 10, train_loss: 0.016201, valid_loss: 0.016507, best_loss: 0.016458, best_loss_epoch: 9
    seed: 4, FOLD: 3, EPOCH: 20, train_loss: 0.013172, valid_loss: 0.015884, best_loss: 0.015874, best_loss_epoch: 19
    seed: 4, FOLD: 3, EPOCH: 24, train_loss: 0.011105, valid_loss: 0.015741, best_loss: 0.015741, best_loss_epoch: 24
    seed: 4, FOLD: 4, EPOCH: 0, train_loss: 0.713596, valid_loss: 0.589952, best_loss: 0.589952, best_loss_epoch: 0
    seed: 4, FOLD: 4, EPOCH: 10, train_loss: 0.016242, valid_loss: 0.016817, best_loss: 0.016747, best_loss_epoch: 9
    seed: 4, FOLD: 4, EPOCH: 20, train_loss: 0.013142, valid_loss: 0.015808, best_loss: 0.015808, best_loss_epoch: 20
    seed: 4, FOLD: 4, EPOCH: 24, train_loss: 0.010845, valid_loss: 0.015817, best_loss: 0.015744, best_loss_epoch: 21
    elapsed time: 911.6501789093018



```python
train.to_pickle(f"{INT_DIR}/{NB}-train-score-pred.pkl")
test.to_pickle(f"{INT_DIR}/{NB}-test-score-pred.pkl")
```


```python
len(target_cols)
```




    206




```python
train[target_cols] = np.maximum(PMIN, np.minimum(PMAX, train[target_cols]))

valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

y_true = train_targets_scored[target_cols].values
y_true = y_true > 0.5
y_pred = valid_results[target_cols].values

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]
    
print("CV log_loss: ", score)
```

    CV log_loss:  0.014315660821881269


- CV log_loss:  0.014761779358699672
- CV log_loss:  0.014519859174255039
- CV log_loss:  0.014525173864593479
- CV log_loss:  0.014354930596928602 # 3 umap features
- CV log_loss:  0.014353604854355429 # more umap features
- CV log_loss:  0.01436484670778641 # more hidden nodes
- CV log_loss:  0.014344688083211073
  - using predicted unscored targets as feature 
- CV log_loss:  0.013368097791623873
  - using given unscored targets as feature
  - bad in public lb
- CV log_loss:  0.01434373547175235
  - rankgauss predicted unscored targets
- CV log_loss:  0.014346100008158216
  - unscored targets pca/umap
- CV log_loss:  0.014328486629791769
  - NFOLDS=10, Epoch=20
- CV log_loss:  0.014299741080816082
  - NFOLDS=10, Epoch=20, 25
- CV log_loss:  0.014311301224480969
  - NFOLDS=10, Epoch=25
- CV log_loss:  0.01429269446076626
  - NFOLDS=10, Epoch=15, 25


```python
# train = pd.read_pickle(f"../interim/23-train-score-pred.pkl")
# test = pd.read_pickle(f"../interim/23-test-score-pred.pkl")
```


```python
train = pd.read_pickle(f"{INT_DIR}/{NB}-train-score-pred.pkl")
test = pd.read_pickle(f"{INT_DIR}/{NB}-test-score-pred.pkl")
```


```python
EPOCHS = 25
# NFOLDS = 5
```


```python
PMIN = 0.0005
PMAX = 0.9995
for c in train_targets_scored.columns:
    if c != "sig_id":
        train_targets_scored[c] = np.maximum(PMIN, np.minimum(PMAX, train_targets_scored[c]))
```


```python
train_targets_scored.columns
```




    Index(['sig_id', '5-alpha_reductase_inhibitor', '11-beta-hsd1_inhibitor',
           'acat_inhibitor', 'acetylcholine_receptor_agonist',
           'acetylcholine_receptor_antagonist', 'acetylcholinesterase_inhibitor',
           'adenosine_receptor_agonist', 'adenosine_receptor_antagonist',
           'adenylyl_cyclase_activator',
           ...
           'tropomyosin_receptor_kinase_inhibitor', 'trpv_agonist',
           'trpv_antagonist', 'tubulin_inhibitor', 'tyrosine_kinase_inhibitor',
           'ubiquitin_specific_protease_inhibitor', 'vegfr_inhibitor', 'vitamin_b',
           'vitamin_d_receptor_agonist', 'wnt_inhibitor'],
          dtype='object', length=207)




```python
train = train[train_targets_scored.columns]
train.columns = [c + "_pred" if (c != 'sig_id' and c in train_targets_scored.columns) else c for c in train.columns]
```


```python
test = test[train_targets_scored.columns]
test.columns = [c + "_pred" if (c != 'sig_id' and c in train_targets_scored.columns) else c for c in test.columns]
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>5-alpha_reductase_inhibitor_pred</th>
      <th>11-beta-hsd1_inhibitor_pred</th>
      <th>acat_inhibitor_pred</th>
      <th>acetylcholine_receptor_agonist_pred</th>
      <th>acetylcholine_receptor_antagonist_pred</th>
      <th>acetylcholinesterase_inhibitor_pred</th>
      <th>adenosine_receptor_agonist_pred</th>
      <th>adenosine_receptor_antagonist_pred</th>
      <th>adenylyl_cyclase_activator_pred</th>
      <th>...</th>
      <th>tropomyosin_receptor_kinase_inhibitor_pred</th>
      <th>trpv_agonist_pred</th>
      <th>trpv_antagonist_pred</th>
      <th>tubulin_inhibitor_pred</th>
      <th>tyrosine_kinase_inhibitor_pred</th>
      <th>ubiquitin_specific_protease_inhibitor_pred</th>
      <th>vegfr_inhibitor_pred</th>
      <th>vitamin_b_pred</th>
      <th>vitamin_d_receptor_agonist_pred</th>
      <th>wnt_inhibitor_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>0.001109</td>
      <td>0.000566</td>
      <td>0.001031</td>
      <td>0.013340</td>
      <td>0.022308</td>
      <td>0.009981</td>
      <td>0.009920</td>
      <td>0.001822</td>
      <td>0.000356</td>
      <td>...</td>
      <td>0.000401</td>
      <td>0.000432</td>
      <td>0.002294</td>
      <td>0.001227</td>
      <td>0.001260</td>
      <td>0.000308</td>
      <td>0.000786</td>
      <td>0.001496</td>
      <td>0.000272</td>
      <td>0.002475</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>0.000517</td>
      <td>0.000717</td>
      <td>0.001716</td>
      <td>0.022324</td>
      <td>0.015328</td>
      <td>0.005460</td>
      <td>0.003773</td>
      <td>0.006745</td>
      <td>0.000360</td>
      <td>...</td>
      <td>0.000652</td>
      <td>0.000992</td>
      <td>0.001912</td>
      <td>0.003241</td>
      <td>0.001773</td>
      <td>0.000353</td>
      <td>0.001338</td>
      <td>0.003170</td>
      <td>0.000739</td>
      <td>0.002624</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_000a6266a</td>
      <td>0.001134</td>
      <td>0.001519</td>
      <td>0.001242</td>
      <td>0.003622</td>
      <td>0.005393</td>
      <td>0.002933</td>
      <td>0.001191</td>
      <td>0.007574</td>
      <td>0.000914</td>
      <td>...</td>
      <td>0.000215</td>
      <td>0.002115</td>
      <td>0.001481</td>
      <td>0.003549</td>
      <td>0.023159</td>
      <td>0.000632</td>
      <td>0.193047</td>
      <td>0.001601</td>
      <td>0.000388</td>
      <td>0.000964</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_0015fd391</td>
      <td>0.000235</td>
      <td>0.000448</td>
      <td>0.001300</td>
      <td>0.006650</td>
      <td>0.012237</td>
      <td>0.001122</td>
      <td>0.002041</td>
      <td>0.001892</td>
      <td>0.000294</td>
      <td>...</td>
      <td>0.000474</td>
      <td>0.001788</td>
      <td>0.002124</td>
      <td>0.036906</td>
      <td>0.004617</td>
      <td>0.000750</td>
      <td>0.000973</td>
      <td>0.001115</td>
      <td>0.000197</td>
      <td>0.000508</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_001626bd3</td>
      <td>0.000349</td>
      <td>0.000602</td>
      <td>0.003679</td>
      <td>0.010258</td>
      <td>0.016118</td>
      <td>0.001603</td>
      <td>0.006557</td>
      <td>0.002199</td>
      <td>0.000786</td>
      <td>...</td>
      <td>0.001215</td>
      <td>0.001419</td>
      <td>0.004727</td>
      <td>0.003352</td>
      <td>0.002853</td>
      <td>0.000662</td>
      <td>0.001429</td>
      <td>0.002994</td>
      <td>0.000358</td>
      <td>0.001202</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21943</th>
      <td>id_fff8c2444</td>
      <td>0.002457</td>
      <td>0.003158</td>
      <td>0.000500</td>
      <td>0.011378</td>
      <td>0.043455</td>
      <td>0.003157</td>
      <td>0.000989</td>
      <td>0.003326</td>
      <td>0.000184</td>
      <td>...</td>
      <td>0.000440</td>
      <td>0.000454</td>
      <td>0.002706</td>
      <td>0.000574</td>
      <td>0.000537</td>
      <td>0.000818</td>
      <td>0.001636</td>
      <td>0.000928</td>
      <td>0.000728</td>
      <td>0.000424</td>
    </tr>
    <tr>
      <th>21944</th>
      <td>id_fffb1ceed</td>
      <td>0.003395</td>
      <td>0.001667</td>
      <td>0.000633</td>
      <td>0.006318</td>
      <td>0.018109</td>
      <td>0.004784</td>
      <td>0.002684</td>
      <td>0.002405</td>
      <td>0.000224</td>
      <td>...</td>
      <td>0.000332</td>
      <td>0.000277</td>
      <td>0.003392</td>
      <td>0.001393</td>
      <td>0.001225</td>
      <td>0.000408</td>
      <td>0.002090</td>
      <td>0.001274</td>
      <td>0.000327</td>
      <td>0.000914</td>
    </tr>
    <tr>
      <th>21945</th>
      <td>id_fffb70c0c</td>
      <td>0.000487</td>
      <td>0.000423</td>
      <td>0.002047</td>
      <td>0.001332</td>
      <td>0.002610</td>
      <td>0.001505</td>
      <td>0.001868</td>
      <td>0.004940</td>
      <td>0.000665</td>
      <td>...</td>
      <td>0.000278</td>
      <td>0.000591</td>
      <td>0.001572</td>
      <td>0.000175</td>
      <td>0.003707</td>
      <td>0.000349</td>
      <td>0.001984</td>
      <td>0.001426</td>
      <td>0.000882</td>
      <td>0.004799</td>
    </tr>
    <tr>
      <th>21946</th>
      <td>id_fffcb9e7c</td>
      <td>0.000221</td>
      <td>0.000270</td>
      <td>0.000453</td>
      <td>0.000943</td>
      <td>0.002244</td>
      <td>0.000904</td>
      <td>0.000860</td>
      <td>0.000685</td>
      <td>0.000080</td>
      <td>...</td>
      <td>0.000140</td>
      <td>0.000212</td>
      <td>0.000793</td>
      <td>0.001575</td>
      <td>0.001680</td>
      <td>0.000141</td>
      <td>0.001713</td>
      <td>0.000351</td>
      <td>0.000154</td>
      <td>0.000240</td>
    </tr>
    <tr>
      <th>21947</th>
      <td>id_ffffdd77b</td>
      <td>0.000213</td>
      <td>0.000620</td>
      <td>0.000331</td>
      <td>0.000476</td>
      <td>0.002273</td>
      <td>0.001104</td>
      <td>0.000318</td>
      <td>0.000863</td>
      <td>0.000076</td>
      <td>...</td>
      <td>0.000088</td>
      <td>0.001094</td>
      <td>0.000835</td>
      <td>0.009346</td>
      <td>0.000601</td>
      <td>0.001128</td>
      <td>0.000615</td>
      <td>0.000425</td>
      <td>0.000121</td>
      <td>0.000089</td>
    </tr>
  </tbody>
</table>
<p>21948 rows × 207 columns</p>
</div>




```python
# use nonscored target in the given file as feature
# if comment out below, use predicted nonscored target
# train = train.drop(nonscored_target, axis=1)
# train = train.merge(train_targets_nonscored, on="sig_id")
# train = train_features.merge(train_targets_scored, on='sig_id')
train = train.merge(train_targets_scored, on='sig_id')
# train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
# test = test[test['cp_type']!='ctl_vehicle'].reset_index(drop=True)

# target = train[train_targets_scored.columns]
target = train[train_targets_scored.columns]
```


```python
# train["cp_time"] = train_features[train_features["cp_type"]=="trt_cp"].reset_index(drop=True)["cp_time"]
# train["cp_dose"] = train_features[train_features["cp_type"]=="trt_cp"].reset_index(drop=True)["cp_dose"]
# test["cp_time"] = test_features[test_features["cp_type"]=="trt_cp"].reset_index(drop=True)["cp_time"]
# test["cp_dose"] = test_features[test_features["cp_type"]=="trt_cp"].reset_index(drop=True)["cp_dose"]
```


```python
from sklearn.preprocessing import QuantileTransformer

scored_target_pred = [c + "_pred" for c in train_targets_scored.columns if c != 'sig_id']

for col in (scored_target_pred):

#     transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
    vec_len = len(train[col].values)
    vec_len_test = len(test[col].values)
    raw_vec = train[col].values.reshape(vec_len, 1)
#     transformer.fit(raw_vec)
    if IS_TRAIN:
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
        transformer.fit(raw_vec)
        pd.to_pickle(transformer, f"{MODEL_DIR}/{NB}_{col}_quantile_scored.pkl")
    else:
        transformer = pd.read_pickle(f"{MODEL_DIR}/{NB}_{col}_quantile_scored.pkl")

    train[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test[col] = transformer.transform(test[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]
```


```python
# train = train.drop('cp_type', axis=1)
# test = test.drop('cp_type', axis=1)
```


```python
target_cols = target.drop('sig_id', axis=1).columns.values.tolist()
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>5-alpha_reductase_inhibitor_pred</th>
      <th>11-beta-hsd1_inhibitor_pred</th>
      <th>acat_inhibitor_pred</th>
      <th>acetylcholine_receptor_agonist_pred</th>
      <th>acetylcholine_receptor_antagonist_pred</th>
      <th>acetylcholinesterase_inhibitor_pred</th>
      <th>adenosine_receptor_agonist_pred</th>
      <th>adenosine_receptor_antagonist_pred</th>
      <th>adenylyl_cyclase_activator_pred</th>
      <th>...</th>
      <th>tropomyosin_receptor_kinase_inhibitor</th>
      <th>trpv_agonist</th>
      <th>trpv_antagonist</th>
      <th>tubulin_inhibitor</th>
      <th>tyrosine_kinase_inhibitor</th>
      <th>ubiquitin_specific_protease_inhibitor</th>
      <th>vegfr_inhibitor</th>
      <th>vitamin_b</th>
      <th>vitamin_d_receptor_agonist</th>
      <th>wnt_inhibitor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>0.919950</td>
      <td>-0.202092</td>
      <td>-0.077408</td>
      <td>0.735101</td>
      <td>0.725399</td>
      <td>1.814577</td>
      <td>2.084492</td>
      <td>-0.701726</td>
      <td>0.526557</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>-0.092546</td>
      <td>0.112945</td>
      <td>0.869530</td>
      <td>1.430804</td>
      <td>0.142064</td>
      <td>0.732099</td>
      <td>0.919336</td>
      <td>0.984604</td>
      <td>0.543999</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_000a6266a</td>
      <td>0.946596</td>
      <td>1.071957</td>
      <td>0.274303</td>
      <td>-0.599946</td>
      <td>-0.704624</td>
      <td>-0.185996</td>
      <td>-0.517498</td>
      <td>1.125989</td>
      <td>1.706798</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_0015fd391</td>
      <td>-0.975515</td>
      <td>-0.514173</td>
      <td>0.358029</td>
      <td>-0.064000</td>
      <td>-0.121431</td>
      <td>-0.895735</td>
      <td>0.100228</td>
      <td>-0.663960</td>
      <td>0.212917</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_001626bd3</td>
      <td>-0.602819</td>
      <td>-0.117040</td>
      <td>2.128986</td>
      <td>0.411235</td>
      <td>0.209720</td>
      <td>-0.689657</td>
      <td>1.616742</td>
      <td>-0.510589</td>
      <td>1.552811</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21943</th>
      <td>id_fff8c2444</td>
      <td>1.766759</td>
      <td>1.906934</td>
      <td>-1.242759</td>
      <td>0.534047</td>
      <td>1.942313</td>
      <td>-0.100230</td>
      <td>-0.682480</td>
      <td>0.027154</td>
      <td>-0.586623</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>21944</th>
      <td>id_fffb1ceed</td>
      <td>2.074651</td>
      <td>1.187644</td>
      <td>-0.887778</td>
      <td>-0.114399</td>
      <td>0.387096</td>
      <td>0.508680</td>
      <td>0.457073</td>
      <td>-0.402069</td>
      <td>-0.264819</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>21945</th>
      <td>id_fffb70c0c</td>
      <td>-0.173258</td>
      <td>-0.582089</td>
      <td>1.195086</td>
      <td>-1.390102</td>
      <td>-1.159643</td>
      <td>-0.725703</td>
      <td>-0.007766</td>
      <td>0.570568</td>
      <td>1.368370</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>21946</th>
      <td>id_fffcb9e7c</td>
      <td>-1.039416</td>
      <td>-1.021251</td>
      <td>-1.400523</td>
      <td>-1.788361</td>
      <td>-1.297889</td>
      <td>-1.023616</td>
      <td>-0.801925</td>
      <td>-1.371752</td>
      <td>-1.505861</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
    </tr>
    <tr>
      <th>21947</th>
      <td>id_ffffdd77b</td>
      <td>-1.072098</td>
      <td>-0.079571</td>
      <td>-1.899018</td>
      <td>-2.399144</td>
      <td>-1.284667</td>
      <td>-0.905270</td>
      <td>-1.840316</td>
      <td>-1.187313</td>
      <td>-1.553955</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
    </tr>
  </tbody>
</table>
<p>21948 rows × 413 columns</p>
</div>




```python
folds = train.copy()

mskf = MultilabelStratifiedKFold(n_splits=NFOLDS)

for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
    folds.loc[v_idx, 'kfold'] = int(f)

folds['kfold'] = folds['kfold'].astype(int)
folds
```

    /opt/conda/lib/python3.7/site-packages/sklearn/utils/validation.py:70: FutureWarning: Pass shuffle=False, random_state=None as keyword args. From version 0.25 passing these as positional arguments will result in an error
      FutureWarning)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>5-alpha_reductase_inhibitor_pred</th>
      <th>11-beta-hsd1_inhibitor_pred</th>
      <th>acat_inhibitor_pred</th>
      <th>acetylcholine_receptor_agonist_pred</th>
      <th>acetylcholine_receptor_antagonist_pred</th>
      <th>acetylcholinesterase_inhibitor_pred</th>
      <th>adenosine_receptor_agonist_pred</th>
      <th>adenosine_receptor_antagonist_pred</th>
      <th>adenylyl_cyclase_activator_pred</th>
      <th>...</th>
      <th>trpv_agonist</th>
      <th>trpv_antagonist</th>
      <th>tubulin_inhibitor</th>
      <th>tyrosine_kinase_inhibitor</th>
      <th>ubiquitin_specific_protease_inhibitor</th>
      <th>vegfr_inhibitor</th>
      <th>vitamin_b</th>
      <th>vitamin_d_receptor_agonist</th>
      <th>wnt_inhibitor</th>
      <th>kfold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>0.919950</td>
      <td>-0.202092</td>
      <td>-0.077408</td>
      <td>0.735101</td>
      <td>0.725399</td>
      <td>1.814577</td>
      <td>2.084492</td>
      <td>-0.701726</td>
      <td>0.526557</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>-0.092546</td>
      <td>0.112945</td>
      <td>0.869530</td>
      <td>1.430804</td>
      <td>0.142064</td>
      <td>0.732099</td>
      <td>0.919336</td>
      <td>0.984604</td>
      <td>0.543999</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_000a6266a</td>
      <td>0.946596</td>
      <td>1.071957</td>
      <td>0.274303</td>
      <td>-0.599946</td>
      <td>-0.704624</td>
      <td>-0.185996</td>
      <td>-0.517498</td>
      <td>1.125989</td>
      <td>1.706798</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_0015fd391</td>
      <td>-0.975515</td>
      <td>-0.514173</td>
      <td>0.358029</td>
      <td>-0.064000</td>
      <td>-0.121431</td>
      <td>-0.895735</td>
      <td>0.100228</td>
      <td>-0.663960</td>
      <td>0.212917</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_001626bd3</td>
      <td>-0.602819</td>
      <td>-0.117040</td>
      <td>2.128986</td>
      <td>0.411235</td>
      <td>0.209720</td>
      <td>-0.689657</td>
      <td>1.616742</td>
      <td>-0.510589</td>
      <td>1.552811</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21943</th>
      <td>id_fff8c2444</td>
      <td>1.766759</td>
      <td>1.906934</td>
      <td>-1.242759</td>
      <td>0.534047</td>
      <td>1.942313</td>
      <td>-0.100230</td>
      <td>-0.682480</td>
      <td>0.027154</td>
      <td>-0.586623</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21944</th>
      <td>id_fffb1ceed</td>
      <td>2.074651</td>
      <td>1.187644</td>
      <td>-0.887778</td>
      <td>-0.114399</td>
      <td>0.387096</td>
      <td>0.508680</td>
      <td>0.457073</td>
      <td>-0.402069</td>
      <td>-0.264819</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21945</th>
      <td>id_fffb70c0c</td>
      <td>-0.173258</td>
      <td>-0.582089</td>
      <td>1.195086</td>
      <td>-1.390102</td>
      <td>-1.159643</td>
      <td>-0.725703</td>
      <td>-0.007766</td>
      <td>0.570568</td>
      <td>1.368370</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21946</th>
      <td>id_fffcb9e7c</td>
      <td>-1.039416</td>
      <td>-1.021251</td>
      <td>-1.400523</td>
      <td>-1.788361</td>
      <td>-1.297889</td>
      <td>-1.023616</td>
      <td>-0.801925</td>
      <td>-1.371752</td>
      <td>-1.505861</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>4</td>
    </tr>
    <tr>
      <th>21947</th>
      <td>id_ffffdd77b</td>
      <td>-1.072098</td>
      <td>-0.079571</td>
      <td>-1.899018</td>
      <td>-2.399144</td>
      <td>-1.284667</td>
      <td>-0.905270</td>
      <td>-1.840316</td>
      <td>-1.187313</td>
      <td>-1.553955</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>21948 rows × 414 columns</p>
</div>




```python
print(train.shape)
print(folds.shape)
print(test.shape)
print(target.shape)
print(sample_submission.shape)
```

    (21948, 413)
    (21948, 414)
    (3624, 207)
    (21948, 207)
    (3982, 207)



```python
folds
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>5-alpha_reductase_inhibitor_pred</th>
      <th>11-beta-hsd1_inhibitor_pred</th>
      <th>acat_inhibitor_pred</th>
      <th>acetylcholine_receptor_agonist_pred</th>
      <th>acetylcholine_receptor_antagonist_pred</th>
      <th>acetylcholinesterase_inhibitor_pred</th>
      <th>adenosine_receptor_agonist_pred</th>
      <th>adenosine_receptor_antagonist_pred</th>
      <th>adenylyl_cyclase_activator_pred</th>
      <th>...</th>
      <th>trpv_agonist</th>
      <th>trpv_antagonist</th>
      <th>tubulin_inhibitor</th>
      <th>tyrosine_kinase_inhibitor</th>
      <th>ubiquitin_specific_protease_inhibitor</th>
      <th>vegfr_inhibitor</th>
      <th>vitamin_b</th>
      <th>vitamin_d_receptor_agonist</th>
      <th>wnt_inhibitor</th>
      <th>kfold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>0.919950</td>
      <td>-0.202092</td>
      <td>-0.077408</td>
      <td>0.735101</td>
      <td>0.725399</td>
      <td>1.814577</td>
      <td>2.084492</td>
      <td>-0.701726</td>
      <td>0.526557</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>-0.092546</td>
      <td>0.112945</td>
      <td>0.869530</td>
      <td>1.430804</td>
      <td>0.142064</td>
      <td>0.732099</td>
      <td>0.919336</td>
      <td>0.984604</td>
      <td>0.543999</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_000a6266a</td>
      <td>0.946596</td>
      <td>1.071957</td>
      <td>0.274303</td>
      <td>-0.599946</td>
      <td>-0.704624</td>
      <td>-0.185996</td>
      <td>-0.517498</td>
      <td>1.125989</td>
      <td>1.706798</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_0015fd391</td>
      <td>-0.975515</td>
      <td>-0.514173</td>
      <td>0.358029</td>
      <td>-0.064000</td>
      <td>-0.121431</td>
      <td>-0.895735</td>
      <td>0.100228</td>
      <td>-0.663960</td>
      <td>0.212917</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_001626bd3</td>
      <td>-0.602819</td>
      <td>-0.117040</td>
      <td>2.128986</td>
      <td>0.411235</td>
      <td>0.209720</td>
      <td>-0.689657</td>
      <td>1.616742</td>
      <td>-0.510589</td>
      <td>1.552811</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21943</th>
      <td>id_fff8c2444</td>
      <td>1.766759</td>
      <td>1.906934</td>
      <td>-1.242759</td>
      <td>0.534047</td>
      <td>1.942313</td>
      <td>-0.100230</td>
      <td>-0.682480</td>
      <td>0.027154</td>
      <td>-0.586623</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21944</th>
      <td>id_fffb1ceed</td>
      <td>2.074651</td>
      <td>1.187644</td>
      <td>-0.887778</td>
      <td>-0.114399</td>
      <td>0.387096</td>
      <td>0.508680</td>
      <td>0.457073</td>
      <td>-0.402069</td>
      <td>-0.264819</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21945</th>
      <td>id_fffb70c0c</td>
      <td>-0.173258</td>
      <td>-0.582089</td>
      <td>1.195086</td>
      <td>-1.390102</td>
      <td>-1.159643</td>
      <td>-0.725703</td>
      <td>-0.007766</td>
      <td>0.570568</td>
      <td>1.368370</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21946</th>
      <td>id_fffcb9e7c</td>
      <td>-1.039416</td>
      <td>-1.021251</td>
      <td>-1.400523</td>
      <td>-1.788361</td>
      <td>-1.297889</td>
      <td>-1.023616</td>
      <td>-0.801925</td>
      <td>-1.371752</td>
      <td>-1.505861</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>4</td>
    </tr>
    <tr>
      <th>21947</th>
      <td>id_ffffdd77b</td>
      <td>-1.072098</td>
      <td>-0.079571</td>
      <td>-1.899018</td>
      <td>-2.399144</td>
      <td>-1.284667</td>
      <td>-0.905270</td>
      <td>-1.840316</td>
      <td>-1.187313</td>
      <td>-1.553955</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>21948 rows × 414 columns</p>
</div>




```python
def process_data(data):
    
#     data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
#     data.loc[:, 'cp_time'] = data.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2, 0:0, 1:1, 2:2})
#     data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1, 0:0, 1:1})

# --------------------- Normalize ---------------------
#     for col in GENES:
#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))
    
#     for col in CELLS:
#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))
    
#--------------------- Removing Skewness ---------------------
#     for col in GENES + CELLS:
#         if(abs(data[col].skew()) > 0.75):
            
#             if(data[col].skew() < 0): # neg-skewness
#                 data[col] = data[col].max() - data[col] + 1
#                 data[col] = np.sqrt(data[col])
            
#             else:
#                 data[col] = np.sqrt(data[col])
    
    return data
```


```python
feature_cols = [c for c in folds.columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]
len(feature_cols)
```




    206




```python
feature_cols
```




    ['5-alpha_reductase_inhibitor_pred',
     '11-beta-hsd1_inhibitor_pred',
     'acat_inhibitor_pred',
     'acetylcholine_receptor_agonist_pred',
     'acetylcholine_receptor_antagonist_pred',
     'acetylcholinesterase_inhibitor_pred',
     'adenosine_receptor_agonist_pred',
     'adenosine_receptor_antagonist_pred',
     'adenylyl_cyclase_activator_pred',
     'adrenergic_receptor_agonist_pred',
     'adrenergic_receptor_antagonist_pred',
     'akt_inhibitor_pred',
     'aldehyde_dehydrogenase_inhibitor_pred',
     'alk_inhibitor_pred',
     'ampk_activator_pred',
     'analgesic_pred',
     'androgen_receptor_agonist_pred',
     'androgen_receptor_antagonist_pred',
     'anesthetic_-_local_pred',
     'angiogenesis_inhibitor_pred',
     'angiotensin_receptor_antagonist_pred',
     'anti-inflammatory_pred',
     'antiarrhythmic_pred',
     'antibiotic_pred',
     'anticonvulsant_pred',
     'antifungal_pred',
     'antihistamine_pred',
     'antimalarial_pred',
     'antioxidant_pred',
     'antiprotozoal_pred',
     'antiviral_pred',
     'apoptosis_stimulant_pred',
     'aromatase_inhibitor_pred',
     'atm_kinase_inhibitor_pred',
     'atp-sensitive_potassium_channel_antagonist_pred',
     'atp_synthase_inhibitor_pred',
     'atpase_inhibitor_pred',
     'atr_kinase_inhibitor_pred',
     'aurora_kinase_inhibitor_pred',
     'autotaxin_inhibitor_pred',
     'bacterial_30s_ribosomal_subunit_inhibitor_pred',
     'bacterial_50s_ribosomal_subunit_inhibitor_pred',
     'bacterial_antifolate_pred',
     'bacterial_cell_wall_synthesis_inhibitor_pred',
     'bacterial_dna_gyrase_inhibitor_pred',
     'bacterial_dna_inhibitor_pred',
     'bacterial_membrane_integrity_inhibitor_pred',
     'bcl_inhibitor_pred',
     'bcr-abl_inhibitor_pred',
     'benzodiazepine_receptor_agonist_pred',
     'beta_amyloid_inhibitor_pred',
     'bromodomain_inhibitor_pred',
     'btk_inhibitor_pred',
     'calcineurin_inhibitor_pred',
     'calcium_channel_blocker_pred',
     'cannabinoid_receptor_agonist_pred',
     'cannabinoid_receptor_antagonist_pred',
     'carbonic_anhydrase_inhibitor_pred',
     'casein_kinase_inhibitor_pred',
     'caspase_activator_pred',
     'catechol_o_methyltransferase_inhibitor_pred',
     'cc_chemokine_receptor_antagonist_pred',
     'cck_receptor_antagonist_pred',
     'cdk_inhibitor_pred',
     'chelating_agent_pred',
     'chk_inhibitor_pred',
     'chloride_channel_blocker_pred',
     'cholesterol_inhibitor_pred',
     'cholinergic_receptor_antagonist_pred',
     'coagulation_factor_inhibitor_pred',
     'corticosteroid_agonist_pred',
     'cyclooxygenase_inhibitor_pred',
     'cytochrome_p450_inhibitor_pred',
     'dihydrofolate_reductase_inhibitor_pred',
     'dipeptidyl_peptidase_inhibitor_pred',
     'diuretic_pred',
     'dna_alkylating_agent_pred',
     'dna_inhibitor_pred',
     'dopamine_receptor_agonist_pred',
     'dopamine_receptor_antagonist_pred',
     'egfr_inhibitor_pred',
     'elastase_inhibitor_pred',
     'erbb2_inhibitor_pred',
     'estrogen_receptor_agonist_pred',
     'estrogen_receptor_antagonist_pred',
     'faah_inhibitor_pred',
     'farnesyltransferase_inhibitor_pred',
     'fatty_acid_receptor_agonist_pred',
     'fgfr_inhibitor_pred',
     'flt3_inhibitor_pred',
     'focal_adhesion_kinase_inhibitor_pred',
     'free_radical_scavenger_pred',
     'fungal_squalene_epoxidase_inhibitor_pred',
     'gaba_receptor_agonist_pred',
     'gaba_receptor_antagonist_pred',
     'gamma_secretase_inhibitor_pred',
     'glucocorticoid_receptor_agonist_pred',
     'glutamate_inhibitor_pred',
     'glutamate_receptor_agonist_pred',
     'glutamate_receptor_antagonist_pred',
     'gonadotropin_receptor_agonist_pred',
     'gsk_inhibitor_pred',
     'hcv_inhibitor_pred',
     'hdac_inhibitor_pred',
     'histamine_receptor_agonist_pred',
     'histamine_receptor_antagonist_pred',
     'histone_lysine_demethylase_inhibitor_pred',
     'histone_lysine_methyltransferase_inhibitor_pred',
     'hiv_inhibitor_pred',
     'hmgcr_inhibitor_pred',
     'hsp_inhibitor_pred',
     'igf-1_inhibitor_pred',
     'ikk_inhibitor_pred',
     'imidazoline_receptor_agonist_pred',
     'immunosuppressant_pred',
     'insulin_secretagogue_pred',
     'insulin_sensitizer_pred',
     'integrin_inhibitor_pred',
     'jak_inhibitor_pred',
     'kit_inhibitor_pred',
     'laxative_pred',
     'leukotriene_inhibitor_pred',
     'leukotriene_receptor_antagonist_pred',
     'lipase_inhibitor_pred',
     'lipoxygenase_inhibitor_pred',
     'lxr_agonist_pred',
     'mdm_inhibitor_pred',
     'mek_inhibitor_pred',
     'membrane_integrity_inhibitor_pred',
     'mineralocorticoid_receptor_antagonist_pred',
     'monoacylglycerol_lipase_inhibitor_pred',
     'monoamine_oxidase_inhibitor_pred',
     'monopolar_spindle_1_kinase_inhibitor_pred',
     'mtor_inhibitor_pred',
     'mucolytic_agent_pred',
     'neuropeptide_receptor_antagonist_pred',
     'nfkb_inhibitor_pred',
     'nicotinic_receptor_agonist_pred',
     'nitric_oxide_donor_pred',
     'nitric_oxide_production_inhibitor_pred',
     'nitric_oxide_synthase_inhibitor_pred',
     'norepinephrine_reuptake_inhibitor_pred',
     'nrf2_activator_pred',
     'opioid_receptor_agonist_pred',
     'opioid_receptor_antagonist_pred',
     'orexin_receptor_antagonist_pred',
     'p38_mapk_inhibitor_pred',
     'p-glycoprotein_inhibitor_pred',
     'parp_inhibitor_pred',
     'pdgfr_inhibitor_pred',
     'pdk_inhibitor_pred',
     'phosphodiesterase_inhibitor_pred',
     'phospholipase_inhibitor_pred',
     'pi3k_inhibitor_pred',
     'pkc_inhibitor_pred',
     'potassium_channel_activator_pred',
     'potassium_channel_antagonist_pred',
     'ppar_receptor_agonist_pred',
     'ppar_receptor_antagonist_pred',
     'progesterone_receptor_agonist_pred',
     'progesterone_receptor_antagonist_pred',
     'prostaglandin_inhibitor_pred',
     'prostanoid_receptor_antagonist_pred',
     'proteasome_inhibitor_pred',
     'protein_kinase_inhibitor_pred',
     'protein_phosphatase_inhibitor_pred',
     'protein_synthesis_inhibitor_pred',
     'protein_tyrosine_kinase_inhibitor_pred',
     'radiopaque_medium_pred',
     'raf_inhibitor_pred',
     'ras_gtpase_inhibitor_pred',
     'retinoid_receptor_agonist_pred',
     'retinoid_receptor_antagonist_pred',
     'rho_associated_kinase_inhibitor_pred',
     'ribonucleoside_reductase_inhibitor_pred',
     'rna_polymerase_inhibitor_pred',
     'serotonin_receptor_agonist_pred',
     'serotonin_receptor_antagonist_pred',
     'serotonin_reuptake_inhibitor_pred',
     'sigma_receptor_agonist_pred',
     'sigma_receptor_antagonist_pred',
     'smoothened_receptor_antagonist_pred',
     'sodium_channel_inhibitor_pred',
     'sphingosine_receptor_agonist_pred',
     'src_inhibitor_pred',
     'steroid_pred',
     'syk_inhibitor_pred',
     'tachykinin_antagonist_pred',
     'tgf-beta_receptor_inhibitor_pred',
     'thrombin_inhibitor_pred',
     'thymidylate_synthase_inhibitor_pred',
     'tlr_agonist_pred',
     'tlr_antagonist_pred',
     'tnf_inhibitor_pred',
     'topoisomerase_inhibitor_pred',
     'transient_receptor_potential_channel_antagonist_pred',
     'tropomyosin_receptor_kinase_inhibitor_pred',
     'trpv_agonist_pred',
     'trpv_antagonist_pred',
     'tubulin_inhibitor_pred',
     'tyrosine_kinase_inhibitor_pred',
     'ubiquitin_specific_protease_inhibitor_pred',
     'vegfr_inhibitor_pred',
     'vitamin_b_pred',
     'vitamin_d_receptor_agonist_pred',
     'wnt_inhibitor_pred']




```python
folds
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>5-alpha_reductase_inhibitor_pred</th>
      <th>11-beta-hsd1_inhibitor_pred</th>
      <th>acat_inhibitor_pred</th>
      <th>acetylcholine_receptor_agonist_pred</th>
      <th>acetylcholine_receptor_antagonist_pred</th>
      <th>acetylcholinesterase_inhibitor_pred</th>
      <th>adenosine_receptor_agonist_pred</th>
      <th>adenosine_receptor_antagonist_pred</th>
      <th>adenylyl_cyclase_activator_pred</th>
      <th>...</th>
      <th>trpv_agonist</th>
      <th>trpv_antagonist</th>
      <th>tubulin_inhibitor</th>
      <th>tyrosine_kinase_inhibitor</th>
      <th>ubiquitin_specific_protease_inhibitor</th>
      <th>vegfr_inhibitor</th>
      <th>vitamin_b</th>
      <th>vitamin_d_receptor_agonist</th>
      <th>wnt_inhibitor</th>
      <th>kfold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_000644bb2</td>
      <td>0.919950</td>
      <td>-0.202092</td>
      <td>-0.077408</td>
      <td>0.735101</td>
      <td>0.725399</td>
      <td>1.814577</td>
      <td>2.084492</td>
      <td>-0.701726</td>
      <td>0.526557</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_000779bfc</td>
      <td>-0.092546</td>
      <td>0.112945</td>
      <td>0.869530</td>
      <td>1.430804</td>
      <td>0.142064</td>
      <td>0.732099</td>
      <td>0.919336</td>
      <td>0.984604</td>
      <td>0.543999</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_000a6266a</td>
      <td>0.946596</td>
      <td>1.071957</td>
      <td>0.274303</td>
      <td>-0.599946</td>
      <td>-0.704624</td>
      <td>-0.185996</td>
      <td>-0.517498</td>
      <td>1.125989</td>
      <td>1.706798</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_0015fd391</td>
      <td>-0.975515</td>
      <td>-0.514173</td>
      <td>0.358029</td>
      <td>-0.064000</td>
      <td>-0.121431</td>
      <td>-0.895735</td>
      <td>0.100228</td>
      <td>-0.663960</td>
      <td>0.212917</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_001626bd3</td>
      <td>-0.602819</td>
      <td>-0.117040</td>
      <td>2.128986</td>
      <td>0.411235</td>
      <td>0.209720</td>
      <td>-0.689657</td>
      <td>1.616742</td>
      <td>-0.510589</td>
      <td>1.552811</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21943</th>
      <td>id_fff8c2444</td>
      <td>1.766759</td>
      <td>1.906934</td>
      <td>-1.242759</td>
      <td>0.534047</td>
      <td>1.942313</td>
      <td>-0.100230</td>
      <td>-0.682480</td>
      <td>0.027154</td>
      <td>-0.586623</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21944</th>
      <td>id_fffb1ceed</td>
      <td>2.074651</td>
      <td>1.187644</td>
      <td>-0.887778</td>
      <td>-0.114399</td>
      <td>0.387096</td>
      <td>0.508680</td>
      <td>0.457073</td>
      <td>-0.402069</td>
      <td>-0.264819</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21945</th>
      <td>id_fffb70c0c</td>
      <td>-0.173258</td>
      <td>-0.582089</td>
      <td>1.195086</td>
      <td>-1.390102</td>
      <td>-1.159643</td>
      <td>-0.725703</td>
      <td>-0.007766</td>
      <td>0.570568</td>
      <td>1.368370</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21946</th>
      <td>id_fffcb9e7c</td>
      <td>-1.039416</td>
      <td>-1.021251</td>
      <td>-1.400523</td>
      <td>-1.788361</td>
      <td>-1.297889</td>
      <td>-1.023616</td>
      <td>-0.801925</td>
      <td>-1.371752</td>
      <td>-1.505861</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>4</td>
    </tr>
    <tr>
      <th>21947</th>
      <td>id_ffffdd77b</td>
      <td>-1.072098</td>
      <td>-0.079571</td>
      <td>-1.899018</td>
      <td>-2.399144</td>
      <td>-1.284667</td>
      <td>-0.905270</td>
      <td>-1.840316</td>
      <td>-1.187313</td>
      <td>-1.553955</td>
      <td>...</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>0.0005</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>21948 rows × 414 columns</p>
</div>




```python
EPOCHS = 25
num_features=len(feature_cols)
num_targets=len(target_cols)
hidden_size=1024
# hidden_size=4096
# hidden_size=9192
```


```python
def run_training(fold, seed):
    
    seed_everything(seed)
    
    train = process_data(folds)
    test_ = process_data(test)
    
    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index
    
    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)
    
    x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values
    
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.3, div_factor=1000, 
#                                               max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.2, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    loss_fn = nn.BCEWithLogitsLoss()
    
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
    
    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    best_loss = np.inf
    best_loss_epoch = -1
    
    if IS_TRAIN:
        for epoch in range(EPOCHS):

            train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)

            if valid_loss < best_loss:            
                best_loss = valid_loss
                best_loss_epoch = epoch
                oof[val_idx] = valid_preds
                torch.save(model.state_dict(), f"{MODEL_DIR}/{NB}-scored2-SEED{seed}-FOLD{fold}_.pth")
            elif(EARLY_STOP == True):
                early_step += 1
                if (early_step >= early_stopping_steps):
                    break

            if epoch % 10 == 0 or epoch == EPOCHS-1:
                print(f"seed: {seed}, FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}, best_loss: {best_loss:.6f}, best_loss_epoch: {best_loss_epoch}")                           
    
    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.load_state_dict(torch.load(f"{MODEL_DIR}/{NB}-scored2-SEED{seed}-FOLD{fold}_.pth"))
    model.to(DEVICE)
    
    if not IS_TRAIN:
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
        oof[val_idx] = valid_preds     
    
    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)
    
    return oof, predictions
```


```python
def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))
    
    for fold in range(NFOLDS):
        oof_, pred_ = run_training(fold, seed)
        
        predictions += pred_ / NFOLDS
        oof += oof_
        
    return oof, predictions
```


```python
SEED = range(NSEEDS)  # [0, 1, 2, 3 ,4]#, 5, 6, 7, 8, 9, 10]
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

time_start = time.time()

for seed in SEED:
    
    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)
    print(f"elapsed time: {time.time() - time_start}")

train[target_cols] = oof
test[target_cols] = predictions
```

    seed: 0, FOLD: 0, EPOCH: 0, train_loss: 0.718041, valid_loss: 0.648379, best_loss: 0.648379, best_loss_epoch: 0
    seed: 0, FOLD: 0, EPOCH: 10, train_loss: 0.019157, valid_loss: 0.019229, best_loss: 0.019229, best_loss_epoch: 10
    seed: 0, FOLD: 0, EPOCH: 20, train_loss: 0.018175, valid_loss: 0.018835, best_loss: 0.018835, best_loss_epoch: 20
    seed: 0, FOLD: 0, EPOCH: 24, train_loss: 0.017637, valid_loss: 0.018741, best_loss: 0.018741, best_loss_epoch: 24
    seed: 0, FOLD: 1, EPOCH: 0, train_loss: 0.718309, valid_loss: 0.640266, best_loss: 0.640266, best_loss_epoch: 0
    seed: 0, FOLD: 1, EPOCH: 10, train_loss: 0.019163, valid_loss: 0.019598, best_loss: 0.019585, best_loss_epoch: 9
    seed: 0, FOLD: 1, EPOCH: 20, train_loss: 0.018238, valid_loss: 0.019068, best_loss: 0.019068, best_loss_epoch: 20
    seed: 0, FOLD: 1, EPOCH: 24, train_loss: 0.017658, valid_loss: 0.018958, best_loss: 0.018958, best_loss_epoch: 24
    seed: 0, FOLD: 2, EPOCH: 0, train_loss: 0.718461, valid_loss: 0.643918, best_loss: 0.643918, best_loss_epoch: 0
    seed: 0, FOLD: 2, EPOCH: 10, train_loss: 0.019301, valid_loss: 0.018816, best_loss: 0.018816, best_loss_epoch: 10
    seed: 0, FOLD: 2, EPOCH: 20, train_loss: 0.018321, valid_loss: 0.018347, best_loss: 0.018347, best_loss_epoch: 20
    seed: 0, FOLD: 2, EPOCH: 24, train_loss: 0.017818, valid_loss: 0.018232, best_loss: 0.018224, best_loss_epoch: 23
    seed: 0, FOLD: 3, EPOCH: 0, train_loss: 0.718506, valid_loss: 0.642388, best_loss: 0.642388, best_loss_epoch: 0
    seed: 0, FOLD: 3, EPOCH: 10, train_loss: 0.019215, valid_loss: 0.019232, best_loss: 0.019221, best_loss_epoch: 9
    seed: 0, FOLD: 3, EPOCH: 20, train_loss: 0.018251, valid_loss: 0.018669, best_loss: 0.018669, best_loss_epoch: 20
    seed: 0, FOLD: 3, EPOCH: 24, train_loss: 0.017705, valid_loss: 0.018556, best_loss: 0.018556, best_loss_epoch: 24
    seed: 0, FOLD: 4, EPOCH: 0, train_loss: 0.718434, valid_loss: 0.641242, best_loss: 0.641242, best_loss_epoch: 0
    seed: 0, FOLD: 4, EPOCH: 10, train_loss: 0.019138, valid_loss: 0.019465, best_loss: 0.019465, best_loss_epoch: 10
    seed: 0, FOLD: 4, EPOCH: 20, train_loss: 0.018215, valid_loss: 0.019114, best_loss: 0.019114, best_loss_epoch: 20
    seed: 0, FOLD: 4, EPOCH: 24, train_loss: 0.017668, valid_loss: 0.018986, best_loss: 0.018980, best_loss_epoch: 22
    elapsed time: 107.86585021018982
    seed: 1, FOLD: 0, EPOCH: 0, train_loss: 0.720025, valid_loss: 0.650747, best_loss: 0.650747, best_loss_epoch: 0
    seed: 1, FOLD: 0, EPOCH: 10, train_loss: 0.019182, valid_loss: 0.019465, best_loss: 0.019316, best_loss_epoch: 9
    seed: 1, FOLD: 0, EPOCH: 20, train_loss: 0.018203, valid_loss: 0.018839, best_loss: 0.018839, best_loss_epoch: 20
    seed: 1, FOLD: 0, EPOCH: 24, train_loss: 0.017714, valid_loss: 0.018737, best_loss: 0.018737, best_loss_epoch: 24
    seed: 1, FOLD: 1, EPOCH: 0, train_loss: 0.719552, valid_loss: 0.637456, best_loss: 0.637456, best_loss_epoch: 0
    seed: 1, FOLD: 1, EPOCH: 10, train_loss: 0.019142, valid_loss: 0.019623, best_loss: 0.019578, best_loss_epoch: 9
    seed: 1, FOLD: 1, EPOCH: 20, train_loss: 0.018179, valid_loss: 0.019100, best_loss: 0.019100, best_loss_epoch: 20
    seed: 1, FOLD: 1, EPOCH: 24, train_loss: 0.017706, valid_loss: 0.018962, best_loss: 0.018962, best_loss_epoch: 24
    seed: 1, FOLD: 2, EPOCH: 0, train_loss: 0.719755, valid_loss: 0.645522, best_loss: 0.645522, best_loss_epoch: 0
    seed: 1, FOLD: 2, EPOCH: 10, train_loss: 0.019294, valid_loss: 0.018932, best_loss: 0.018899, best_loss_epoch: 8
    seed: 1, FOLD: 2, EPOCH: 20, train_loss: 0.018312, valid_loss: 0.018321, best_loss: 0.018321, best_loss_epoch: 20
    seed: 1, FOLD: 2, EPOCH: 24, train_loss: 0.017800, valid_loss: 0.018244, best_loss: 0.018244, best_loss_epoch: 24
    seed: 1, FOLD: 3, EPOCH: 0, train_loss: 0.720024, valid_loss: 0.642908, best_loss: 0.642908, best_loss_epoch: 0
    seed: 1, FOLD: 3, EPOCH: 10, train_loss: 0.019222, valid_loss: 0.019299, best_loss: 0.019223, best_loss_epoch: 9
    seed: 1, FOLD: 3, EPOCH: 20, train_loss: 0.018277, valid_loss: 0.018663, best_loss: 0.018663, best_loss_epoch: 20
    seed: 1, FOLD: 3, EPOCH: 24, train_loss: 0.017741, valid_loss: 0.018568, best_loss: 0.018566, best_loss_epoch: 23
    seed: 1, FOLD: 4, EPOCH: 0, train_loss: 0.719621, valid_loss: 0.645065, best_loss: 0.645065, best_loss_epoch: 0
    seed: 1, FOLD: 4, EPOCH: 10, train_loss: 0.019146, valid_loss: 0.019653, best_loss: 0.019541, best_loss_epoch: 8
    seed: 1, FOLD: 4, EPOCH: 20, train_loss: 0.018164, valid_loss: 0.019035, best_loss: 0.019035, best_loss_epoch: 20
    seed: 1, FOLD: 4, EPOCH: 24, train_loss: 0.017654, valid_loss: 0.018938, best_loss: 0.018938, best_loss_epoch: 24
    elapsed time: 215.6553032398224
    seed: 2, FOLD: 0, EPOCH: 0, train_loss: 0.719241, valid_loss: 0.645240, best_loss: 0.645240, best_loss_epoch: 0
    seed: 2, FOLD: 0, EPOCH: 10, train_loss: 0.019185, valid_loss: 0.019330, best_loss: 0.019317, best_loss_epoch: 9
    seed: 2, FOLD: 0, EPOCH: 20, train_loss: 0.018206, valid_loss: 0.018788, best_loss: 0.018788, best_loss_epoch: 20
    seed: 2, FOLD: 0, EPOCH: 24, train_loss: 0.017682, valid_loss: 0.018714, best_loss: 0.018714, best_loss_epoch: 24
    seed: 2, FOLD: 1, EPOCH: 0, train_loss: 0.719620, valid_loss: 0.636283, best_loss: 0.636283, best_loss_epoch: 0
    seed: 2, FOLD: 1, EPOCH: 10, train_loss: 0.019157, valid_loss: 0.019578, best_loss: 0.019578, best_loss_epoch: 10
    seed: 2, FOLD: 1, EPOCH: 20, train_loss: 0.018216, valid_loss: 0.019064, best_loss: 0.019064, best_loss_epoch: 20
    seed: 2, FOLD: 1, EPOCH: 24, train_loss: 0.017701, valid_loss: 0.018972, best_loss: 0.018965, best_loss_epoch: 22
    seed: 2, FOLD: 2, EPOCH: 0, train_loss: 0.719221, valid_loss: 0.645173, best_loss: 0.645173, best_loss_epoch: 0
    seed: 2, FOLD: 2, EPOCH: 10, train_loss: 0.019312, valid_loss: 0.018955, best_loss: 0.018754, best_loss_epoch: 9
    seed: 2, FOLD: 2, EPOCH: 20, train_loss: 0.018335, valid_loss: 0.018361, best_loss: 0.018361, best_loss_epoch: 20
    seed: 2, FOLD: 2, EPOCH: 24, train_loss: 0.017821, valid_loss: 0.018264, best_loss: 0.018264, best_loss_epoch: 24
    seed: 2, FOLD: 3, EPOCH: 0, train_loss: 0.719251, valid_loss: 0.638802, best_loss: 0.638802, best_loss_epoch: 0
    seed: 2, FOLD: 3, EPOCH: 10, train_loss: 0.019248, valid_loss: 0.019212, best_loss: 0.019212, best_loss_epoch: 10
    seed: 2, FOLD: 3, EPOCH: 20, train_loss: 0.018260, valid_loss: 0.018666, best_loss: 0.018666, best_loss_epoch: 20
    seed: 2, FOLD: 3, EPOCH: 24, train_loss: 0.017719, valid_loss: 0.018537, best_loss: 0.018537, best_loss_epoch: 24
    seed: 2, FOLD: 4, EPOCH: 0, train_loss: 0.719448, valid_loss: 0.643739, best_loss: 0.643739, best_loss_epoch: 0
    seed: 2, FOLD: 4, EPOCH: 10, train_loss: 0.019167, valid_loss: 0.019611, best_loss: 0.019511, best_loss_epoch: 9
    seed: 2, FOLD: 4, EPOCH: 20, train_loss: 0.018196, valid_loss: 0.019099, best_loss: 0.019099, best_loss_epoch: 20
    seed: 2, FOLD: 4, EPOCH: 24, train_loss: 0.017700, valid_loss: 0.018942, best_loss: 0.018941, best_loss_epoch: 23
    elapsed time: 323.71297216415405
    seed: 3, FOLD: 0, EPOCH: 0, train_loss: 0.719928, valid_loss: 0.644308, best_loss: 0.644308, best_loss_epoch: 0
    seed: 3, FOLD: 0, EPOCH: 10, train_loss: 0.019177, valid_loss: 0.019441, best_loss: 0.019373, best_loss_epoch: 8
    seed: 3, FOLD: 0, EPOCH: 20, train_loss: 0.018225, valid_loss: 0.018795, best_loss: 0.018795, best_loss_epoch: 20
    seed: 3, FOLD: 0, EPOCH: 24, train_loss: 0.017688, valid_loss: 0.018718, best_loss: 0.018718, best_loss_epoch: 24
    seed: 3, FOLD: 1, EPOCH: 0, train_loss: 0.720076, valid_loss: 0.647205, best_loss: 0.647205, best_loss_epoch: 0
    seed: 3, FOLD: 1, EPOCH: 10, train_loss: 0.019234, valid_loss: 0.019587, best_loss: 0.019587, best_loss_epoch: 10
    seed: 3, FOLD: 1, EPOCH: 20, train_loss: 0.018210, valid_loss: 0.019078, best_loss: 0.019078, best_loss_epoch: 20
    seed: 3, FOLD: 1, EPOCH: 24, train_loss: 0.017711, valid_loss: 0.018969, best_loss: 0.018969, best_loss_epoch: 24
    seed: 3, FOLD: 2, EPOCH: 0, train_loss: 0.719672, valid_loss: 0.639483, best_loss: 0.639483, best_loss_epoch: 0
    seed: 3, FOLD: 2, EPOCH: 10, train_loss: 0.019304, valid_loss: 0.018939, best_loss: 0.018938, best_loss_epoch: 8
    seed: 3, FOLD: 2, EPOCH: 20, train_loss: 0.018316, valid_loss: 0.018311, best_loss: 0.018311, best_loss_epoch: 20
    seed: 3, FOLD: 2, EPOCH: 24, train_loss: 0.017793, valid_loss: 0.018231, best_loss: 0.018219, best_loss_epoch: 23
    seed: 3, FOLD: 3, EPOCH: 0, train_loss: 0.720188, valid_loss: 0.637864, best_loss: 0.637864, best_loss_epoch: 0
    seed: 3, FOLD: 3, EPOCH: 10, train_loss: 0.019247, valid_loss: 0.019216, best_loss: 0.019191, best_loss_epoch: 9
    seed: 3, FOLD: 3, EPOCH: 20, train_loss: 0.018206, valid_loss: 0.018631, best_loss: 0.018631, best_loss_epoch: 20
    seed: 3, FOLD: 3, EPOCH: 24, train_loss: 0.017732, valid_loss: 0.018547, best_loss: 0.018547, best_loss_epoch: 24
    seed: 3, FOLD: 4, EPOCH: 0, train_loss: 0.719878, valid_loss: 0.638939, best_loss: 0.638939, best_loss_epoch: 0
    seed: 3, FOLD: 4, EPOCH: 10, train_loss: 0.019190, valid_loss: 0.019601, best_loss: 0.019563, best_loss_epoch: 9
    seed: 3, FOLD: 4, EPOCH: 20, train_loss: 0.018175, valid_loss: 0.019032, best_loss: 0.019032, best_loss_epoch: 20
    seed: 3, FOLD: 4, EPOCH: 24, train_loss: 0.017663, valid_loss: 0.018959, best_loss: 0.018959, best_loss_epoch: 24
    elapsed time: 431.6321220397949
    seed: 4, FOLD: 0, EPOCH: 0, train_loss: 0.719533, valid_loss: 0.642937, best_loss: 0.642937, best_loss_epoch: 0
    seed: 4, FOLD: 0, EPOCH: 10, train_loss: 0.019206, valid_loss: 0.019308, best_loss: 0.019293, best_loss_epoch: 9
    seed: 4, FOLD: 0, EPOCH: 20, train_loss: 0.018242, valid_loss: 0.018796, best_loss: 0.018796, best_loss_epoch: 20
    seed: 4, FOLD: 0, EPOCH: 24, train_loss: 0.017664, valid_loss: 0.018721, best_loss: 0.018721, best_loss_epoch: 24
    seed: 4, FOLD: 1, EPOCH: 0, train_loss: 0.719307, valid_loss: 0.643715, best_loss: 0.643715, best_loss_epoch: 0
    seed: 4, FOLD: 1, EPOCH: 10, train_loss: 0.019151, valid_loss: 0.019547, best_loss: 0.019493, best_loss_epoch: 9
    seed: 4, FOLD: 1, EPOCH: 20, train_loss: 0.018214, valid_loss: 0.019096, best_loss: 0.019096, best_loss_epoch: 20
    seed: 4, FOLD: 1, EPOCH: 24, train_loss: 0.017681, valid_loss: 0.018967, best_loss: 0.018967, best_loss_epoch: 24
    seed: 4, FOLD: 2, EPOCH: 0, train_loss: 0.719541, valid_loss: 0.642515, best_loss: 0.642515, best_loss_epoch: 0
    seed: 4, FOLD: 2, EPOCH: 10, train_loss: 0.019343, valid_loss: 0.018892, best_loss: 0.018892, best_loss_epoch: 10
    seed: 4, FOLD: 2, EPOCH: 20, train_loss: 0.018368, valid_loss: 0.018344, best_loss: 0.018344, best_loss_epoch: 20
    seed: 4, FOLD: 2, EPOCH: 24, train_loss: 0.017845, valid_loss: 0.018234, best_loss: 0.018231, best_loss_epoch: 23
    seed: 4, FOLD: 3, EPOCH: 0, train_loss: 0.719370, valid_loss: 0.637958, best_loss: 0.637958, best_loss_epoch: 0
    seed: 4, FOLD: 3, EPOCH: 10, train_loss: 0.019256, valid_loss: 0.019176, best_loss: 0.019176, best_loss_epoch: 10
    seed: 4, FOLD: 3, EPOCH: 20, train_loss: 0.018263, valid_loss: 0.018699, best_loss: 0.018699, best_loss_epoch: 20
    seed: 4, FOLD: 3, EPOCH: 24, train_loss: 0.017779, valid_loss: 0.018567, best_loss: 0.018567, best_loss_epoch: 24
    seed: 4, FOLD: 4, EPOCH: 0, train_loss: 0.719547, valid_loss: 0.641542, best_loss: 0.641542, best_loss_epoch: 0
    seed: 4, FOLD: 4, EPOCH: 10, train_loss: 0.019106, valid_loss: 0.019679, best_loss: 0.019608, best_loss_epoch: 8
    seed: 4, FOLD: 4, EPOCH: 20, train_loss: 0.018217, valid_loss: 0.019049, best_loss: 0.019049, best_loss_epoch: 20
    seed: 4, FOLD: 4, EPOCH: 24, train_loss: 0.017681, valid_loss: 0.018946, best_loss: 0.018946, best_loss_epoch: 24
    elapsed time: 538.5660672187805



```python
train.to_pickle(f"{INT_DIR}/{NB}-train-score-stack-pred.pkl")
test.to_pickle(f"{INT_DIR}/{NB}-test-score-stack-pred.pkl")
```


```python
train[target_cols] = np.maximum(PMIN, np.minimum(PMAX, train[target_cols]))
valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)

y_true = train_targets_scored[target_cols].values
y_true = y_true > 0.5
y_pred = valid_results[target_cols].values

y_pred = np.minimum(SMAX, np.maximum(SMIN, y_pred))

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]
    
print("CV log_loss: ", score)
```

    CV log_loss:  0.014317539872155836



```python
# for c in test.columns:
#     if c != "sig_id":
#         test[c] = np.maximum(PMIN, np.minimum(PMAX, test[c]))

sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
sub.to_csv('submission_kibuna_nn.csv', index=False)
```


```python
sub
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sig_id</th>
      <th>5-alpha_reductase_inhibitor</th>
      <th>11-beta-hsd1_inhibitor</th>
      <th>acat_inhibitor</th>
      <th>acetylcholine_receptor_agonist</th>
      <th>acetylcholine_receptor_antagonist</th>
      <th>acetylcholinesterase_inhibitor</th>
      <th>adenosine_receptor_agonist</th>
      <th>adenosine_receptor_antagonist</th>
      <th>adenylyl_cyclase_activator</th>
      <th>...</th>
      <th>tropomyosin_receptor_kinase_inhibitor</th>
      <th>trpv_agonist</th>
      <th>trpv_antagonist</th>
      <th>tubulin_inhibitor</th>
      <th>tyrosine_kinase_inhibitor</th>
      <th>ubiquitin_specific_protease_inhibitor</th>
      <th>vegfr_inhibitor</th>
      <th>vitamin_b</th>
      <th>vitamin_d_receptor_agonist</th>
      <th>wnt_inhibitor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id_0004d9e33</td>
      <td>0.001162</td>
      <td>0.001199</td>
      <td>0.002015</td>
      <td>0.015341</td>
      <td>0.019063</td>
      <td>0.005686</td>
      <td>0.002673</td>
      <td>0.004332</td>
      <td>0.000621</td>
      <td>...</td>
      <td>0.000948</td>
      <td>0.001240</td>
      <td>0.002639</td>
      <td>0.001860</td>
      <td>0.001598</td>
      <td>0.000875</td>
      <td>0.002036</td>
      <td>0.002026</td>
      <td>0.002357</td>
      <td>0.001651</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id_001897cda</td>
      <td>0.000544</td>
      <td>0.001825</td>
      <td>0.001908</td>
      <td>0.004645</td>
      <td>0.002973</td>
      <td>0.002454</td>
      <td>0.008670</td>
      <td>0.013257</td>
      <td>0.005185</td>
      <td>...</td>
      <td>0.001420</td>
      <td>0.000769</td>
      <td>0.007042</td>
      <td>0.000692</td>
      <td>0.009463</td>
      <td>0.000563</td>
      <td>0.003897</td>
      <td>0.000919</td>
      <td>0.001955</td>
      <td>0.002435</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id_002429b5b</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id_00276f245</td>
      <td>0.001352</td>
      <td>0.001203</td>
      <td>0.001915</td>
      <td>0.014601</td>
      <td>0.023381</td>
      <td>0.005119</td>
      <td>0.002313</td>
      <td>0.004537</td>
      <td>0.000656</td>
      <td>...</td>
      <td>0.000814</td>
      <td>0.001161</td>
      <td>0.003990</td>
      <td>0.005968</td>
      <td>0.007616</td>
      <td>0.000940</td>
      <td>0.001803</td>
      <td>0.002189</td>
      <td>0.000965</td>
      <td>0.003710</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id_0027f1083</td>
      <td>0.001711</td>
      <td>0.002068</td>
      <td>0.002662</td>
      <td>0.012916</td>
      <td>0.025770</td>
      <td>0.004339</td>
      <td>0.005071</td>
      <td>0.004173</td>
      <td>0.000761</td>
      <td>...</td>
      <td>0.001369</td>
      <td>0.001007</td>
      <td>0.004916</td>
      <td>0.002865</td>
      <td>0.002082</td>
      <td>0.001042</td>
      <td>0.002457</td>
      <td>0.002503</td>
      <td>0.000773</td>
      <td>0.001412</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3977</th>
      <td>id_ff7004b87</td>
      <td>0.001349</td>
      <td>0.001697</td>
      <td>0.001539</td>
      <td>0.004341</td>
      <td>0.008089</td>
      <td>0.003488</td>
      <td>0.001529</td>
      <td>0.002775</td>
      <td>0.000657</td>
      <td>...</td>
      <td>0.000746</td>
      <td>0.003383</td>
      <td>0.003631</td>
      <td>0.191109</td>
      <td>0.005271</td>
      <td>0.001207</td>
      <td>0.003119</td>
      <td>0.001709</td>
      <td>0.000754</td>
      <td>0.001466</td>
    </tr>
    <tr>
      <th>3978</th>
      <td>id_ff925dd0d</td>
      <td>0.003415</td>
      <td>0.002313</td>
      <td>0.001492</td>
      <td>0.012155</td>
      <td>0.020872</td>
      <td>0.004502</td>
      <td>0.004942</td>
      <td>0.004304</td>
      <td>0.000844</td>
      <td>...</td>
      <td>0.001008</td>
      <td>0.000954</td>
      <td>0.003261</td>
      <td>0.003732</td>
      <td>0.002666</td>
      <td>0.001236</td>
      <td>0.004973</td>
      <td>0.003352</td>
      <td>0.000814</td>
      <td>0.002385</td>
    </tr>
    <tr>
      <th>3979</th>
      <td>id_ffb710450</td>
      <td>0.001282</td>
      <td>0.001527</td>
      <td>0.002106</td>
      <td>0.014006</td>
      <td>0.028604</td>
      <td>0.005770</td>
      <td>0.004076</td>
      <td>0.005097</td>
      <td>0.000747</td>
      <td>...</td>
      <td>0.000973</td>
      <td>0.001482</td>
      <td>0.003349</td>
      <td>0.003052</td>
      <td>0.002758</td>
      <td>0.000887</td>
      <td>0.002021</td>
      <td>0.001880</td>
      <td>0.001012</td>
      <td>0.001554</td>
    </tr>
    <tr>
      <th>3980</th>
      <td>id_ffbb869f2</td>
      <td>0.002329</td>
      <td>0.001550</td>
      <td>0.001505</td>
      <td>0.019208</td>
      <td>0.020052</td>
      <td>0.005855</td>
      <td>0.007776</td>
      <td>0.002902</td>
      <td>0.000700</td>
      <td>...</td>
      <td>0.000707</td>
      <td>0.000874</td>
      <td>0.003931</td>
      <td>0.002396</td>
      <td>0.001655</td>
      <td>0.000856</td>
      <td>0.001932</td>
      <td>0.002266</td>
      <td>0.000888</td>
      <td>0.003404</td>
    </tr>
    <tr>
      <th>3981</th>
      <td>id_ffd5800b6</td>
      <td>0.000968</td>
      <td>0.001507</td>
      <td>0.001285</td>
      <td>0.008174</td>
      <td>0.016547</td>
      <td>0.005969</td>
      <td>0.001918</td>
      <td>0.003877</td>
      <td>0.000549</td>
      <td>...</td>
      <td>0.000791</td>
      <td>0.001530</td>
      <td>0.002709</td>
      <td>0.004172</td>
      <td>0.002195</td>
      <td>0.001122</td>
      <td>0.001654</td>
      <td>0.002013</td>
      <td>0.000949</td>
      <td>0.001169</td>
    </tr>
  </tbody>
</table>
<p>3982 rows × 207 columns</p>
</div>




```python

```


```python

```


```python

```


```python

```
