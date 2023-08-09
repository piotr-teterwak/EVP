import copy
import datetime
import random
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.datasets import StanfordCars

from util.tool import topk

from time import sleep

from .train_util import ClassSampler

NUM_CLASSES = 196

HOLDOUT_CLASS_IDX = [
 [145, 52, 173, 182, 116, 166, 177, 66, 39, 81],
 [67, 63, 144, 195, 18, 19, 117, 186, 34, 114],
 [78, 137, 119, 131, 63, 11, 103, 108, 93, 21],
 [131, 5, 77, 89, 3, 78, 60, 155, 165, 149],
 [169, 83, 77, 5, 17, 32, 75, 2, 95, 63],
 [116, 38, 25, 62, 147, 189, 44, 170, 176, 145],
 [83, 95, 123, 170, 96, 50, 110, 132, 165, 143],
 [12, 103, 49, 194, 157, 88, 136, 56, 51, 76],
 [186, 171, 89, 66, 1, 20, 62, 61, 141, 195],
 [163, 51, 47, 176, 38, 76, 125, 45, 191, 118]]


class ClassSampledCars(torch.utils.data.Dataset):
  def __init__(self, ds):
    super(ClassSampledCars,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(NUM_CLASSES)]

    for i, (_, label) in enumerate(self.ds._samples):
      self.class_indices[label].append(i)


  def classes(self):
    return self.class_indices

  def __getitem__(self, index):
    return self.ds[index]

  def __len__(self):
      return len(self.ds)

class ClassHoldoutCars(torch.utils.data.Dataset):
  def __init__(self, ds, class_idx):
    super(ClassHoldoutCars,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(NUM_CLASSES)]

    for i, (_,label) in enumerate(self.ds._samples):
      if label not in class_idx:
        self.class_indices[label].append(i)

    self.class_indices = [c for c in self.class_indices if c != []]


  def classes(self):
    return self.class_indices

  def __getitem__(self, index):
    return self.ds[index]

  def __len__(self):
      return sum([len(c) for c in self.class_indices])


class ClassSubsampledCars(torch.utils.data.Dataset):
    def __init__(self, ds, class_idx):
        super(ClassSubsampledCars,self).__init__()
        self.ds = ds
        self.class_idx = class_idx

        self.idx = []

        for i, (_, label) in enumerate(self.ds._samples):
            if label in self.class_idx:
                self.idx.append(i)

    def __getitem__(self, index):
       return self.ds[self.idx[index]]

    def __len__(self):
         return len(self.idx)



def get_cars_data(data_path, batch_size,num_classes_per_batch, preprocess, preprocess_test, num_workers, holdout_class_idx=None, holdout_train=False):
    if holdout_class_idx > -1:
        class_idx = HOLDOUT_CLASS_IDX[holdout_class_idx]
    else:
        class_idx = list(range(NUM_CLASSES))

    test_ds = ClassSubsampledCars(StanfordCars(root=data_path, transform = preprocess_test, split='test'), class_idx=class_idx)
    train_dl = None
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=False, num_workers=num_workers)

    #if holdout_class_idx is not None:
    if holdout_train:
        train_ds = ClassHoldoutCars(StanfordCars(root=data_path, transform = preprocess, split='train'), class_idx=class_idx)	
    else:
        train_ds = ClassSampledCars(StanfordCars(root=data_path, transform = preprocess_test, split='train'))	
    train_sampler = ClassSampler(train_ds.classes(), num_classes_per_batch ,batch_size)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_sampler = train_sampler, pin_memory=True, num_workers=num_workers)
    #train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=num_workers)


    return (train_dl, test_dl)


