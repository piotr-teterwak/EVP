import copy
import datetime
import random
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.datasets import DTD


from .train_util import ClassSampler

NUM_CLASSES = 47

HOLDOUT_CLASS_IDX = [
 [10, 17, 44, 12, 33, 16, 26, 22, 45, 21],
 [43, 14, 4, 41, 27, 6, 37, 11, 2, 5],
 [12, 43, 3, 10, 2, 19, 44, 15, 33, 21],
 [2, 3, 25, 13, 0, 20, 34, 31, 12, 18],
 [39, 45, 10, 6, 25, 33, 30, 4, 12, 46],
 [24, 21, 1, 22, 45, 17, 11, 4, 10, 41],
 [30, 40, 10, 35, 23, 24, 31, 45, 21, 20],
 [29, 2, 4, 44, 1, 0, 7, 26, 28, 36],
 [7, 14, 13, 34, 3, 32, 15, 29, 44, 36],
 [11, 26, 19, 37, 28, 27, 12, 21, 46, 33]]

class ClassSampledDTD(torch.utils.data.Dataset):
  def __init__(self, ds):
    super(ClassSampledDTD,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(NUM_CLASSES)]

    for i, label in enumerate(self.ds._labels):
      self.class_indices[label].append(i)


  def classes(self):
    return self.class_indices

  def __getitem__(self, index):
    return self.ds[index]

  def __len__(self):
      return len(self.ds)

class ClassHoldoutDTD(torch.utils.data.Dataset):
  def __init__(self, ds, class_idx):
    super(ClassHoldoutDTD,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(NUM_CLASSES)]

    for i, label in enumerate(self.ds._labels):
      if label not in class_idx:
        self.class_indices[label].append(i)

    self.class_indices = [c for c in self.class_indices if c != []]


  def classes(self):
    return self.class_indices

  def __getitem__(self, index):
    return self.ds[index]

  def __len__(self):
      return sum([len(c) for c in self.class_indices])


class ClassSubsampledDTD(torch.utils.data.Dataset):
    def __init__(self, ds, class_idx):
        super(ClassSubsampledDTD,self).__init__()
        self.ds = ds
        self.class_idx = class_idx

        self.idx = []

        for i, label in enumerate(self.ds._labels):
            if label in self.class_idx:
                self.idx.append(i)

    def __getitem__(self, index):
       return self.ds[self.idx[index]]

    def __len__(self):
         return len(self.idx)



def get_dtd_data(data_path, batch_size,num_classes_per_batch, preprocess, preprocess_test, num_workers, holdout_class_idx=None, holdout_train=False):
    if holdout_class_idx > -1:
        class_idx = HOLDOUT_CLASS_IDX[holdout_class_idx]
    else:
        class_idx = list(range(NUM_CLASSES))
 
    test_ds = ClassSubsampledDTD(DTD(root=data_path, transform = preprocess_test, split='test'),class_idx=class_idx)
    train_dl = None
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=False, num_workers=num_workers)

    #if holdout_class_idx is not None:
    if holdout_train:
        train_ds = ClassHoldoutDTD(DTD(root=data_path, transform = preprocess, split='train'), class_idx=class_idx)	
    else:
        train_ds = ClassSampledDTD(DTD(root=data_path, transform = preprocess_test, split='train'))	
    train_sampler = ClassSampler(train_ds.classes(), num_classes_per_batch ,batch_size)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_sampler = train_sampler, pin_memory=True, num_workers=num_workers)
    #train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=num_workers)


    return (train_dl, test_dl)


