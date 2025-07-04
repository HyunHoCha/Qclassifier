import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import random
import sys
torch.set_default_dtype(torch.float64)


def extract(dataset, fraction=0.05):
    dataset_list = list(dataset)
    sample_size = int(len(dataset_list) * fraction)
    sampled_data = random.sample(dataset_list, sample_size)

    xs, ys = [], []
    for img, label in sampled_data:
        if label in (3, 6):
            xs.append(img.view(-1).numpy())
            ys.append(0 if label == 3 else 1)
    
    if xs:
        return np.stack(xs), np.array(ys)
    else:
        return np.empty((0, 784)), np.empty((0,))
