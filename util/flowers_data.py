import copy
import datetime
import random
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.datasets import Flowers102

from util.tool import topk

from time import sleep

from .train_util import ClassSampler

HOLDOUT_CLASS_IDX = [
 [76, 70, 101, 9, 52, 96, 10, 59, 82, 24],
 [4, 56, 58, 24, 80, 0, 57, 55, 40, 72],
 [51, 16, 41, 50, 97, 47, 89, 33, 34, 56],
 [89, 74, 46, 57, 62, 56, 45, 5, 96, 22],
 [3, 97, 58, 9, 12, 21, 62, 98, 71, 41],
 [74, 64, 100, 50, 57, 21, 13, 49, 0, 91],
 [45, 58, 41, 34, 101, 47, 99, 42, 69, 28],
 [97, 94, 87, 15, 39, 42, 91, 51, 54, 9],
 [63, 71, 33, 64, 41, 77, 97, 98, 44, 31],
 [61, 22, 15, 95, 14, 16, 79, 19, 84, 32]]

flowers_class_map = {0: 'pink primrose',
 1: 'hard-leaved pocket orchid',
 2: 'canterbury bells',
 3: 'sweet pea',
 4: 'english marigold',
 5: 'tiger lily',
 6: 'moon orchid',
 7: 'bird of paradise',
 8: 'monkshood',
 9: 'globe thistle',
 10: 'snapdragon',
 11: "colt's foot",
 12: 'king protea',
 13: 'spear thistle',
 14: 'yellow iris',
 15: 'globe-flower',
 16: 'purple coneflower',
 17: 'peruvian lily',
 18: 'balloon flower',
 19: 'giant white arum lily',
 20: 'fire lily',
 21: 'pincushion flower',
 22: 'fritillary',
 23: 'red ginger',
 24: 'grape hyacinth',
 25: 'corn poppy',
 26: 'prince of wales feathers',
 27: 'stemless gentian',
 28: 'artichoke',
 29: 'sweet william',
 30: 'carnation',
 31: 'garden phlox',
 32: 'love in the mist',
 33: 'mexican aster',
 34: 'alpine sea holly',
 35: 'ruby-lipped cattleya',
 36: 'cape flower',
 37: 'great masterwort',
 38: 'siam tulip',
 39: 'lenten rose',
 40: 'barbeton daisy',
 41: 'daffodil',
 42: 'sword lily',
 43: 'poinsettia',
 44: 'bolero deep blue',
 45: 'wallflower',
 46: 'marigold',
 47: 'buttercup',
 48: 'oxeye daisy',
 49: 'common dandelion',
 50: 'petunia',
 51: 'wild pansy',
 52: 'primula',
 53: 'sunflower',
 54: 'pelargonium',
 55: 'bishop of llandaff',
 56: 'gaura',
 57: 'geranium',
 58: 'orange dahlia',
 59: 'pink-yellow dahlia?',
 60: 'cautleya spicata',
 61: 'japanese anemone',
 62: 'black-eyed susan',
 63: 'silverbush',
 64: 'californian poppy',
 65: 'osteospermum',
 66: 'spring crocus',
 67: 'bearded iris',
 68: 'windflower',
 69: 'tree poppy',
 70: 'gazania',
 71: 'azalea',
 72: 'water lily',
 73: 'rose',
 74: 'thorn apple',
 75: 'morning glory',
 76: 'passion flower',
 77: 'lotus',
 78: 'toad lily',
 79: 'anthurium',
 80: 'frangipani',
 81: 'clematis',
 82: 'hibiscus',
 83: 'columbine',
 84: 'desert-rose',
 85: 'tree mallow',
 86: 'magnolia',
 87: 'cyclamen ',
 88: 'watercress',
 89: 'canna lily',
 90: 'hippeastrum ',
 91: 'bee balm',
 92: 'ball moss',
 93: 'foxglove',
 94: 'bougainvillea',
 95: 'camellia',
 96: 'mallow',
 97: 'mexican petunia',
 98: 'bromelia',
 99: 'blanket flower',
 100: 'trumpet creeper',
 101: 'blackberry lily'}


class ClassSampledFlowers(torch.utils.data.Dataset):
  def __init__(self, ds):
    super(ClassSampledFlowers,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(102)]

    for i, label in enumerate(self.ds._labels):
      self.class_indices[label].append(i)


  def classes(self):
    return self.class_indices

  def __getitem__(self, index):
    return self.ds[index]

  def __len__(self):
      return len(self.ds)

class ClassHoldoutFlowers(torch.utils.data.Dataset):
  def __init__(self, ds, class_idx):
    super(ClassHoldoutFlowers,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(102)]

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


class ClassSubsampledFlowers(torch.utils.data.Dataset):
    def __init__(self, ds, class_idx):
        super(ClassSubsampledFlowers,self).__init__()
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



def get_flowers_data(data_path, batch_size,num_classes_per_batch, preprocess, preprocess_test, num_workers, holdout_class_idx=None, holdout_train=False):
    if holdout_class_idx > -1:
        class_idx = HOLDOUT_CLASS_IDX[holdout_class_idx]
    else:
        class_idx = list(range(102))
 
    test_ds = ClassSubsampledFlowers(Flowers102(root=data_path, transform = preprocess_test, split='test'), class_idx=class_idx)
    train_dl = None
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=False, num_workers=num_workers)

    #if holdout_class_idx is not None:
    if holdout_train:
        train_ds = ClassHoldoutFlowers(Flowers102(root=data_path, transform = preprocess, split='train'), class_idx=class_idx)	
    else:
        train_ds = ClassSampledFlowers(Flowers102(root=data_path, transform = preprocess_test, split='train'))	
    train_sampler = ClassSampler(train_ds.classes(), num_classes_per_batch ,batch_size)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_sampler = train_sampler, pin_memory=True, num_workers=num_workers)
    #train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=num_workers)


    return (train_dl, test_dl)


