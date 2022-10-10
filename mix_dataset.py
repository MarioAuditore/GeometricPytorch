'''
=======================================
--- Dataset class with mixed labels ---
=======================================
'''
# Our custom will be based on the one from torch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
# Basic libraries for maths
import numpy as np


# Dataset with mixing ability
class MixDataset(Dataset):
    def __init__(self, X, y, n_outsiders, transform = None):
        self.X = X
        self.y = y
        self.transform = transform
        self.n_outsiders = n_outsiders
        
    def __len__(self):
        return len(self.y)
    
    def mix(self, idx):
        X_copy = torch.tensor(self.X[idx]).clone()
    
        # Choose labels to add
        outsider_labels_idx = np.random.randint(0, len(self.y), self.n_outsiders)
        # Check if the amount is right and there are only other labels
        while (np.unique(outsider_labels_idx).shape[0] != self.n_outsiders) or (idx in outsider_labels_idx):
            outsider_labels_idx = np.random.randint(0, len(self.y), self.n_outsiders)
    
        # Choose locations inside set
        outsiders_idx = np.random.randint(0, self.X[idx].shape[0], self.n_outsiders)
        # Check if we have right amount
        while np.unique(outsiders_idx).shape[0] != self.n_outsiders:
            outsiders_idx = np.random.randint(0, self.X[idx].shape[0], self.n_outsiders)
            
            
        for i, label_idx in enumerate(outsider_labels_idx):
            X_copy[outsiders_idx[i], :] = self.X[label_idx, outsiders_idx[i], :]
            
        return X_copy, self.y[idx]

    def __getitem__(self, idx):
        if self.n_outsiders > 0:
            X, y = self.mix(idx)
        else:
            X, y = self.X[idx], self.y[idx]
            
        if self.transform:
            return self.transform(X), y
        return X, y


# When training models we will be dealing with dataloaders
def generate_dataloader(X, y, 
                        transform = None, 
                        n_samples = 6, 
                        one_hot = True, 
                        mix_labels_ratio = 0.2,
                        train_split_ratio = 0.7,
                        batch_size = 32
                       ):
    '''
        n_samples: int
                   Length of one set
    '''
    n_outsiders = int(mix_labels_ratio * n_samples)
    
    if isinstance(y, torch.Tensor):
        labels = y.unique
    elif isinstance(y, np.ndarray):
        labels = np.unique(y)
    
    min_len = len(y)
    for i in labels:
        if min_len > len(y[y == i]):
            min_len = len(y[y == i])
    
    while min_len % n_samples != 0:
        min_len -= 1
    
    X_grouped = np.zeros((len(labels), min_len, *X.shape[1:]))
            
    for i in range(len(labels)):
        X_grouped[i] = X[np.where(y == labels[i])][:min_len]
    
    X_batch = torch.tensor(X_grouped.reshape(len(labels) * int(min_len / n_samples), n_samples, *X.shape[1:]).astype(np.float32))
    y_batch = torch.zeros(len(labels) * int(min_len / n_samples))
    
    for i in range(len(labels)):
        y_batch[int(min_len / n_samples) * i : int(min_len / n_samples) * (i + 1)] = i
        
    if one_hot:
        y_batch = nn.functional.one_hot(y_batch.long()).double()
    
    #dataset = MixDataset(X_batch.swapaxes(2,1), y_batch, n_outsiders = n_outsiders, transform = transform)
    dataset = MixDataset(X_batch, y_batch, n_outsiders = n_outsiders, transform = transform)

    dataset_train, dataset_test = random_split(dataset, [int(train_split_ratio * len(dataset)), len(dataset) - int(train_split_ratio * len(dataset))], generator=torch.Generator())

    dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
    dataloader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = True)
    
    return dataloader_train, dataloader_test
