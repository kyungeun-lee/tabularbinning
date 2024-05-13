import logging
from tqdm import tqdm
import time

from typing import Any, Dict
from collections import defaultdict, OrderedDict
from scipy.io import arff
import numpy as np
import pandas as pd
import os, torchvision, torch, zero
import sklearn.model_selection
import sklearn.datasets
import torch.nn.functional as F
import openml

class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""
    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

            
def get_batch_size(n): 
    ### n = train data size
    if n > 50000:
        return 1024
    elif n > 10000:
        return 512
    elif n > 5000:
        return 256
    elif n > 1000:
        return 128
    else:
        return 64
            

def replace_cat_to_num(datacol):
    items = sorted(datacol.unique().tolist())
    l = len(items)
    return datacol.replace(items, np.arange(l))


def CH(root_dir):
    dt = pd.read_csv(os.path.join(root_dir, "CH", "Churn_Modelling.csv"), index_col=0).iloc[:, 2:]
    for k, v in dict(dt.dtypes).items():
        if not v in ['int64', 'float64']:
            dt[k] = replace_cat_to_num(dt[k])
            
    return {"X_all": dt.drop(['Exited'], axis=1).values,
            "y_all": dt['Exited'].values.reshape([-1, 1]),
            "tasktype": "binclass",
            "columns": dt.columns[:-1], "targetname": "Exited"}


def load_dataset(dataname, device, cat_threshold=20, root_dir="../dataset/", seed=123456):
    
    zero.improve_reproducibility(seed=seed)
    
    dataset = eval(dataname)(root_dir)
    
    dataset['X_train'], dataset['X_test'], dataset['y_train'], dataset['y_test'] = sklearn.model_selection.train_test_split(
        dataset['X_all'].astype('float32'), 
        dataset['y_all'].astype('float32'),
        train_size=0.9
    )
    dataset.pop("X_all"); dataset.pop("y_all")
    dataset['X_train'], dataset['X_val'], dataset['y_train'], dataset['y_val'] = sklearn.model_selection.train_test_split(
        dataset['X_train'], 
        dataset['y_train'], 
        train_size=0.9
    )
    
    dataset['num_features'] = len(dataset['columns'])
    dataset["counts"] = np.array([len(np.unique(dataset['X_train'][:, i])) for i in range(dataset['num_features'])])
    if cat_threshold is None:
        dataset["X_cat"] = []
        dataset["X_num"] = np.arange(dataset["num_features"])
    else:
        dataset["X_cat"] = np.where(dataset["counts"] <= cat_threshold)[0].astype(int)
        dataset["X_num"] = np.array([int(i) for i in range(dataset['num_features']) if not i in dataset['X_cat']])
    
    for k in dataset:
        if isinstance(dataset[k], np.ndarray):
            dataset[k] = torch.from_numpy(dataset[k]).to(device)
        elif isinstance(dataset[k], torch.Tensor):
            dataset[k] = dataset[k].to(device)
        
        if type(dataset[k]) == torch.float64:
            dataset[k] = dataset[k].type(torch.float32)
    
    for cat_dim in dataset["X_cat"]:
        unique_values = torch.unique(dataset["X_train"][:, cat_dim]).cpu()
        mapping_values = torch.arange(len(unique_values)).cpu()
        if torch.all(unique_values == mapping_values):
            pass
        else:
            for dt in ["X_train", "X_val", "X_test"]:
                rawdata = torch.clone(dataset[dt][:, cat_dim])
                revdata = torch.zeros(rawdata.size()).type(torch.int64)
                for (k, v) in zip(unique_values, mapping_values):
                    revdata[torch.where(rawdata == k)[0]] = v
                
                dataset[dt][:, cat_dim] = revdata.to(device)
    
    dataset["stats"] = {"x_mean": dataset["X_train"].mean(0),
                        "x_std": dataset["X_train"].std(0),
                        "y_mean": dataset["y_train"].type(torch.float).mean(0),
                        "y_std": dataset["y_train"].type(torch.float).std(0)}
            
    return dataset

def binning_column(dataset, col, num_bins, target):
    traindata = dataset["X_train"][:, col].cpu() ## Use training data only to determine the bin boundaries
    
    if num_bins == np.inf:
        return np.argsort(traindata)
    elif len(torch.unique(traindata)) < num_bins:
        bins = traindata.unique()
        targetdata = dataset[target][:, col].cpu()
        return np.digitize(targetdata, bins=bins[1:], right=False)
    else:
        targetdata = dataset[target][:, col].cpu()
        bins = np.percentile(traindata, np.arange(0, 100, step=100/num_bins))
        
        bins[-1] = np.inf
        return np.digitize(targetdata, bins=bins[1:], right=False)
    

def Binning(dataset, num_bins, device, binning_reg=True):
            
    binned_dataset = dict()
    for target in ["X_train", "X_val", "X_test"]:
        binned_dataset.update({target: []})
        for col in range(dataset['num_features']):
            binned_col = binning_column(dataset, col, num_bins, target)
            binned_dataset[target].append(binned_col)
        binned_dataset[target] = torch.from_numpy(np.stack(binned_dataset[target], axis=-1)).to(device).type(torch.int64)

        if binning_reg: ## Do standardization to bin indices
            binned_dataset[target] = binned_dataset[target].type(torch.float32)
            binned_dataset["stats"] = {'mean': binned_dataset['X_train'].mean(0, keepdim=True)[0], 'std': binned_dataset['X_train'].std(0, keepdim=True)[0]}
            binned_dataset[target] = (binned_dataset[target] - binned_dataset["stats"]["mean"]) / (binned_dataset["stats"]["std"]+1e-10)

    return binned_dataset


def standardization(dataset, y=False):
    def prep(data, mean, std):
        return (data - mean) / std
    
    for col in range(dataset['num_features']):
        if col in dataset["X_num"]:
            dataset['X_train'][:, col] = prep(dataset['X_train'][:, col], mean=dataset['stats']['x_mean'][col], std=dataset['stats']['x_std'][col])
            dataset['X_val'][:, col] = prep(dataset['X_val'][:, col], mean=dataset['stats']['x_mean'][col], std=dataset['stats']['x_std'][col])
            dataset['X_test'][:, col] = prep(dataset['X_test'][:, col], mean=dataset['stats']['x_mean'][col], std=dataset['stats']['x_std'][col])
    
    if y:
        dataset['y_train'] = prep(dataset['y_train'], mean=dataset['stats']['y_mean'], std=dataset['stats']['y_std'])
        dataset['y_val'] = prep(dataset['y_val'], mean=dataset['stats']['y_mean'], std=dataset['stats']['y_std'])
        dataset['y_test'] = prep(dataset['y_test'], mean=dataset['stats']['y_mean'], std=dataset['stats']['y_std'])
    
    return dataset

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, label, transform=None):
        super().__init__()
        self.dataset = dataset
        self.label = label
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.transform is not None:
            x = self.transform({'image': self.dataset[idx], 'mask': None})
        else:
            x = self.dataset[idx]
        
        if isinstance(self.label, list):
            y = []
            for label in self.label:
                y.append(label[idx])
        else:
            y = self.label[idx]
        return x, y

def cat_num_features(dataset, num_bins=1):
    cat_cardinalities = []
    cat_features = dataset['X_cat']
    for c in cat_features:
        cat_cardinalities.append(len(torch.unique(dataset['X_train'][:, c])))
    num_features = dataset['X_num']
        
    return cat_features, cat_cardinalities, num_features

def data_loader(
    dataname, device, transform_func,
    num_bins=1, binning_reg=True, cat_threshold=20, mode="pretrain", #pretrain or validation
    ):
    
    dataset = load_dataset(dataname, device=device, cat_threshold=cat_threshold)
    
    if (mode == "pretrain") & (num_bins > 0):
        dataset = standardization(dataset, y=False) ## we do not use label dataset here
        binned_dataset = Binning(dataset, num_bins=num_bins, device=device, binning_reg=binning_reg)
        
        cat_features, cat_cardinalities, num_features = cat_num_features(dataset, num_bins=num_bins)

        train_dataset = TabularDataset(dataset['X_train'], binned_dataset['X_train'], transform=transform_func)
        batch_size = get_batch_size(train_dataset.__len__())
        train_loader = zero.data.IndexLoader(train_dataset.__len__(), batch_size, device=device)
        val_dataset = TabularDataset(dataset['X_val'], binned_dataset['X_val'], transform=None)
        val_loader = zero.data.IndexLoader(val_dataset.__len__(), batch_size, device=device)
        
        return (train_dataset, train_loader), (val_dataset, val_loader), (cat_features, cat_cardinalities, num_features), batch_size
    
    else:
        if dataset['tasktype'] == "regression":
            dataset = standardization(dataset, y=True)
        else:
            dataset = standardization(dataset, y=False)
        cat_features, cat_cardinalities, num_features = cat_num_features(dataset)
        
        train_dataset = TabularDataset(dataset['X_train'], dataset['y_train'], transform=None)
    
        batch_size = get_batch_size(train_dataset.__len__())
        train_loader = zero.data.IndexLoader(train_dataset.__len__(), batch_size, device=device)
        val_dataset = TabularDataset(dataset['X_val'], dataset['y_val'], transform=None)
        val_loader = zero.data.IndexLoader(val_dataset.__len__(), batch_size, device=device)
        test_dataset = TabularDataset(dataset['X_test'], dataset['y_test'], transform=None)
        test_loader = zero.data.IndexLoader(val_dataset.__len__(), batch_size, device=device)
        
        return (train_dataset, train_loader), (val_dataset, val_loader), (test_dataset, test_loader), (
            cat_features, cat_cardinalities, num_features), batch_size, (dataset['y_train'].size(1), dataset['tasktype'], dataset['stats']['y_std'])

