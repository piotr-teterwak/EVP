import copy
import datetime
import random
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR100

from util.tool import topk

from time import sleep

from .train_util import ClassSampler

NUM_CLASSES = 100


HOLDOUT_CLASS_IDX = [
 [72, 15, 14, 39, 33, 88, 35, 68, 21, 11],
 [86, 19, 71, 3, 28, 30, 6, 90, 68, 56],
 [36, 61, 10, 82, 75, 6, 34, 12, 17, 35],
 [56, 73, 18, 87, 54, 20, 24, 77, 44, 49],
 [4, 35, 43, 76, 5, 30, 38, 87, 3, 72],
 [30, 45, 28, 69, 72, 60, 83, 2, 79, 67],
 [57, 34, 68, 15, 22, 96, 52, 7, 78, 65],
 [48, 96, 24, 39, 74, 60, 29, 53, 28, 41],
 [48, 7, 74, 26, 68, 53, 72, 38, 24, 81],
 [72, 64, 57, 23, 7, 26, 27, 41, 5, 68]]

class ClassSampledCIFAR100(torch.utils.data.Dataset):
  def __init__(self, ds):
    super(ClassSampledCIFAR100,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(NUM_CLASSES)]

    for i, label in enumerate(self.ds.targets):
      self.class_indices[label].append(i)


  def classes(self):
    return self.class_indices

  def __getitem__(self, index):
    return self.ds[index]

  def __len__(self):
      return len(self.ds)

class ClassHoldoutCIFAR100(torch.utils.data.Dataset):
  def __init__(self, ds, class_idx):
    super(ClassHoldoutCIFAR100,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(NUM_CLASSES)]

    for i, label in enumerate(self.ds.targets):
      if label not in class_idx:
        self.class_indices[label].append(i)

    self.class_indices = [c for c in self.class_indices if c != []]


  def classes(self):
    return self.class_indices

  def __getitem__(self, index):
    return self.ds[index]

  def __len__(self):
      return sum([len(c) for c in self.class_indices])


class ClassSubsampledCIFAR100(torch.utils.data.Dataset):
    def __init__(self, ds, class_idx):
        super(ClassSubsampledCIFAR100,self).__init__()
        self.ds = ds
        self.class_idx = class_idx

        self.idx = []

        for i, label in enumerate(self.ds.targets):
            if label in self.class_idx:
                self.idx.append(i)

    def __getitem__(self, index):
       return self.ds[self.idx[index]]

    def __len__(self):
         return len(self.idx)



def get_cifar100_data(data_path, batch_size,num_classes_per_batch, preprocess, preprocess_test, num_workers, holdout_class_idx=None, holdout_train=False):
    #if holdout_class_idx is not None:
    if holdout_class_idx > -1:
        class_idx = HOLDOUT_CLASS_IDX[holdout_class_idx]
    else:
        class_idx = list(range(NUM_CLASSES))

    test_ds = ClassSubsampledCIFAR100(CIFAR100(root=data_path, transform = preprocess_test, train=False), class_idx=class_idx)
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=False, num_workers=num_workers)

    if holdout_train:
        train_ds = ClassHoldoutCIFAR100(CIFAR100(root=data_path, transform = preprocess, train=True), class_idx=class_idx)	
    else:
        train_ds = ClassSampledCIFAR100(CIFAR100(root=data_path, transform = preprocess_test, train=True))	
    train_sampler = ClassSampler(train_ds.classes(), num_classes_per_batch ,batch_size)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_sampler = train_sampler, pin_memory=True, num_workers=num_workers)
    #train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=num_workers)


    return (train_dl, test_dl)


