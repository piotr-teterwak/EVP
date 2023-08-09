import copy
import datetime
import random
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.datasets import EuroSAT


from .train_util import ClassSampler

NUM_CLASSES = 10

HOLDOUT_CLASS_IDX = [
 [1, 7, 6, 5],
 [8, 6, 4, 1],
 [3, 8, 6, 9],
 [5, 4, 1, 6],
 [4, 2, 6, 3],
 [0, 8, 4, 3],
 [2, 9, 7, 8],
 [5, 4, 8, 6],
 [5, 9, 1, 6],
 [7, 4, 1, 0]]

class ClassSampledEuroSAT(torch.utils.data.Dataset):
  def __init__(self, ds):
    super(ClassSampledEuroSAT,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(NUM_CLASSES)]

    for i, idx in enumerate(self.ds.indices):
      _, label = self.ds[i]
      self.class_indices[label].append(i)


  def classes(self):
    return self.class_indices

  def __getitem__(self, index):
    return self.ds[index]

  def __len__(self):
      return len(self.ds)

class ClassHoldoutEuroSAT(torch.utils.data.Dataset):
  def __init__(self, ds, class_idx):
    super(ClassHoldoutEuroSAT,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(NUM_CLASSES)]

    for i, idx in enumerate(self.ds.indices):
      _, label = self.ds[i]
      if label not in class_idx:
        self.class_indices[label].append(i)

    self.class_indices = [c for c in self.class_indices if c != []]


  def classes(self):
    return self.class_indices

  def __getitem__(self, index):
    return self.ds[index]

  def __len__(self):
      return sum([len(c) for c in self.class_indices])


class ClassSubsampledEuroSAT(torch.utils.data.Dataset):
    def __init__(self, ds, class_idx):
        super(ClassSubsampledEuroSAT,self).__init__()
        self.ds = ds
        self.class_idx = class_idx

        self.idx = []

        for i, idx in enumerate(self.ds.indices):
            _, label = self.ds[i]
            if label in self.class_idx:
                self.idx.append(i)

    def __getitem__(self, index):
       return self.ds[self.idx[index]]

    def __len__(self):
         return len(self.idx)



def get_eurosat_data(data_path, batch_size,num_classes_per_batch, preprocess, preprocess_test, num_workers, holdout_class_idx=None, holdout_train=False):
    if holdout_class_idx > -1:
        class_idx = HOLDOUT_CLASS_IDX[holdout_class_idx]
    else:
        class_idx = list(range(NUM_CLASSES))
    generator = torch.Generator().manual_seed(42)
    base_train_ds = EuroSAT(root=data_path, transform = preprocess)
    base_test_ds = EuroSAT(root=data_path, transform = preprocess_test)
    train_ds, _ = torch.utils.data.random_split(base_train_ds, [0.8,0.2], generator=generator)
    generator = torch.Generator().manual_seed(42)
    _, test_ds = torch.utils.data.random_split(base_test_ds, [0.8,0.2], generator=generator)
    test_ds = ClassSubsampledEuroSAT(test_ds, class_idx=class_idx)
    train_dl = None
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=False, num_workers=num_workers)

    #if holdout_class_idx is not None:
    if holdout_train:
        train_ds = ClassHoldoutEuroSAT(train_ds, class_idx=class_idx)	
    else:
        train_ds = ClassSampledEuroSAT(train_ds)	
    train_sampler = ClassSampler(train_ds.classes(), num_classes_per_batch ,batch_size)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_sampler = train_sampler, pin_memory=True, num_workers=num_workers)
    #train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=num_workers)


    return (train_dl, test_dl)


