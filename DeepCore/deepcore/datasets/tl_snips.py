from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import torch


def TL_SNIPS(data_path):
    
    features_train = np.load(os.path.join(data_path, 'features_train.npy'))
    labels_train = np.load(os.path.join(data_path, 'labels_train.npy'))
    features_train = torch.tensor(features_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.long)
    dst_train = TensorDataset(features_train, labels_train)
    
    features_test = np.load(os.path.join(data_path, 'features_test.npy'))
    labels_test = np.load(os.path.join(data_path, 'labels_test.npy'))
    features_test = torch.tensor(features_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)
    dst_test = TensorDataset(features_test, labels_test)
    
    features_val = np.load(os.path.join(data_path, 'features_val.npy'))
    labels_val = np.load(os.path.join(data_path, 'labels_val.npy'))
    features_val = torch.tensor(features_val, dtype=torch.float32)
    labels_val = torch.tensor(labels_val, dtype=torch.long)
    dst_val = TensorDataset(features_val, labels_val)
    
    return dst_train, dst_test, dst_val
