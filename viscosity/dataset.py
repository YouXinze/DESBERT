from __future__ import print_function, division
import functools
import  numpy  as  np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
import functools
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class SMILES_dataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, df, tokenizer):
        self.smiles1 = df['Smiles1']
        self.smiles2 = df['Smiles2']

        self.mol1 = df['molar1']
        self.mol2 = df['molar2']

        self.HA1 = df['HBA Count1']
        self.HA2 = df['HBA Count2']
        self.HD1 = df['HBD Count1']
        self.HD2 = df['HBD Count2']

        t = df['T/K']
        t = np.array(t).reshape(-1, 1)

        scaler = StandardScaler()
        self.t = scaler.fit_transform(t)

        # Use joblib for parallel tokenization with tqdm for progress bar
        self.tokens1 = np.array(
            list(Parallel(n_jobs=10)(
                delayed(tokenizer.encode)(i, max_length=100, truncation=True, padding='max_length')
                for i in tqdm(self.smiles1, desc='Tokenizing SMILES', total=len(self.smiles1))
            ))
        )
        self.tokens2 = np.array(
            list(Parallel(n_jobs=10)(
                delayed(tokenizer.encode)(i, max_length=100, truncation=True, padding='max_length')
                for i in tqdm(self.smiles2, desc='Tokenizing SMILES', total=len(self.smiles2))
            ))
        )


        self.label = df['Log viscosity']

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        # Tokenize the SMILES string to get X (shape should be [100, 1] if padding length is 100)
        X1 = torch.from_numpy(np.asarray(self.tokens1[index]).astype(np.float32))
        X2 = torch.from_numpy(np.asarray(self.tokens2[index]).astype(np.float32))
        y = torch.from_numpy(np.asarray(self.label[index])).float()

        smiles1 = self.smiles1[index]
        smiles2 = self.smiles2[index]
        mol1 = self.mol1[index]
        mol2 = self.mol2[index]

        HA1 = torch.from_numpy(np.asarray(self.HA1[index]).astype(np.float32))
        HA2 = torch.from_numpy(np.asarray(self.HA2[index]).astype(np.float32))
        HD1 = torch.from_numpy(np.asarray(self.HD1[index]).astype(np.float32))
        HD2 = torch.from_numpy(np.asarray(self.HD2[index]).astype(np.float32))

        T = torch.from_numpy(np.asarray(self.t[index])).float()
        # print(y)
        # print(X.shape, y.shape, smiles)

        return (X1, X2, mol1, mol2,HA1, HD1, HA2, HD2, T), y, (smiles1, smiles2)



