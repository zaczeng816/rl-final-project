#!/usr/bin/env python
import numpy as np
from torch.utils.data import Dataset

class BoardDataset(Dataset):
    def __init__(self, dataset): # dataset = np.array of (s, p, v)
        self.X = dataset[:,0]
        self.y_p, self.y_v = dataset[:,1], dataset[:,2]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return np.int64(self.X[idx].transpose(2,0,1)), self.y_p[idx], self.y_v[idx]
