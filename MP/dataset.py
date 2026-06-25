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



class SMILES_dataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, df, tokenizer):
        self.smiles1 = df['Smiles1']
        self.smiles2 = df['Smiles2']

        self.mol1 = df['molar1']
        self.mol2 = df['molar2']

        self.HBA_a_count = df['HBA Count1']
        self.HBD_a_count = df['HBD Count1']
        self.HBA_b_count = df['HBA Count2']
        self.HBD_b_count = df['HBD Count2']
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

        self.label = df['DES melting temperature']
        # self.label = df[[f'sigma_{i}' for i in range(1, 52)]].values  # Ensure it's a numpy array
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        # Tokenize the SMILES string to get X (shape should be [100, 1] if padding length is 100)
        X1 = torch.from_numpy(np.asarray(self.tokens1[index]).astype(np.float32))
        X2 = torch.from_numpy(np.asarray(self.tokens2[index]).astype(np.float32))
        # Ensure y has shape (51,) where each value corresponds to a sigma_i
        y = torch.from_numpy(np.asarray(self.label[index])).float()  # Shape should be [51]

        smiles1 = self.smiles1[index]
        smiles2 = self.smiles2[index]
        mol1 = self.mol1[index]
        mol2 = self.mol2[index]
        HBA_a_count = torch.from_numpy(np.asarray(self.HBA_a_count[index])).float()
        HBD_a_count = torch.from_numpy(np.asarray(self.HBD_a_count[index])).float()
        HBA_b_count = torch.from_numpy(np.asarray(self.HBA_b_count[index])).float()
        HBD_b_count = torch.from_numpy(np.asarray(self.HBD_b_count[index])).float()
        # print(y)
        # print(X.shape, y.shape, smiles)

        return (X1, X2, mol1, mol2, HBA_a_count,HBD_a_count, HBA_b_count,HBD_b_count), y, (smiles1, smiles2, mol1, mol2)



