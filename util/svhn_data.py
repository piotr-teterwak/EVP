import copy
import datetime
import random
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.datasets import SVHN

from util.tool import topk

from time import sleep

from .train_util import ClassSampler

NUM_CLASSES = 10

HOLDOUT_CLASS_IDX = [
 [4, 3, 8, 7],
 [7, 9, 4, 3],
 [7, 4, 3, 1],
 [0, 6, 8, 3],
 [8, 6, 4, 7],
 [6, 2, 4, 9],
 [1, 4, 8, 6],
 [6, 7, 5, 2],
 [3, 7, 9, 6],
 [9, 7, 6, 3]]

class ClassSampledSVHN(torch.utils.data.Dataset):
  def __init__(self, ds):
    super(ClassSampledSVHN,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(NUM_CLASSES)]

    for i, label in enumerate(self.ds.labels):
      self.class_indices[label].append(i)


  def classes(self):
    return self.class_indices

  def __getitem__(self, index):
    return self.ds[index]

  def __len__(self):
      return len(self.ds)

class ClassHoldoutSVHN(torch.utils.data.Dataset):
  def __init__(self, ds, class_idx):
    super(ClassHoldoutSVHN,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(NUM_CLASSES)]

    for i, label in enumerate(self.ds.labels):
      if label not in class_idx:
        self.class_indices[label].append(i)

    self.class_indices = [c for c in self.class_indices if c != []]


  def classes(self):
    return self.class_indices

  def __getitem__(self, index):
    return self.ds[index]

  def __len__(self):
      return sum([len(c) for c in self.class_indices])


class ClassSubsampledSVHN(torch.utils.data.Dataset):
    def __init__(self, ds, class_idx):
        super(ClassSubsampledSVHN,self).__init__()
        self.ds = ds
        self.class_idx = class_idx

        self.idx = []

        for i, label in enumerate(self.ds.labels):
            if label in self.class_idx:
                self.idx.append(i)

    def __getitem__(self, index):
       return self.ds[self.idx[index]]

    def __len__(self):
         return len(self.idx)



def get_svhn_data(data_path, batch_size,num_classes_per_batch, preprocess, preprocess_test, num_workers, holdout_class_idx=None, holdout_train=False):
    #if holdout_class_idx is not None:
    if holdout_class_idx > -1:
        class_idx = HOLDOUT_CLASS_IDX[holdout_class_idx]
    else:
        class_idx = list(range(NUM_CLASSES))

    test_ds = ClassSubsampledSVHN(SVHN(root=data_path, transform = preprocess_test, split='test'),class_idx=class_idx)
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=False, num_workers=num_workers)

    if holdout_train:
        train_ds = ClassHoldoutSVHN(SVHN(root=data_path, transform = preprocess, split='train'), class_idx=class_idx)	
    else:
        train_ds = ClassSampledSVHN(SVHN(root=data_path, transform = preprocess_test, split='train'))	
    train_sampler = ClassSampler(train_ds.classes(), num_classes_per_batch ,batch_size)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_sampler = train_sampler, pin_memory=True, num_workers=num_workers)
    #train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=num_workers)


    return (train_dl, test_dl)


