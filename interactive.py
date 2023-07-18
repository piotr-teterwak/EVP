#github.com/ryanchankh/cifar100coarse
import numpy as np
from torchvision.datasets import CIFAR100



def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

class CIFAR100Coarse(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']

]

data = CIFAR100("/fsx/pteterwak/data/positional_vpt_data")
data.coarse_targets = sparse2coarse(data.targets)
class_inds = [torch.where(dataset.targets == class_idx)[0]
              for class_idx in dataset.class_to_idx.values()]
class_inds = [torch.where(dataset.targets == class_idx)[0]
              for class_idx in data.class_to_idx.values()]
import torch
class_inds = [torch.where(dataset.targets == class_idx)[0]
              for class_idx in data.class_to_idx.values()]
class_inds = [torch.where(data.targets == class_idx)[0]
              for class_idx in data.class_to_idx.values()]
class_inds = [torch.where(data.coarse_targets == class_idx)[0]
              for class_idx in data.class_to_idx.values()]
class_inds = [torch.where(data.coarse_targets == class_idx)[0]
              for class_idx in data.class_to_idx.values()]
class_inds = [torch.where(data.coarse_targets == class_idx)[0]
              for class_idx in range(len(sparse2coarse))]
sparse2coarse
sparse2coarse()
data.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
data
data.class_to_idx
data.class_to_idx = {v,k for k,v in data.values}
data.coarse_class_to_idx = {v,k for k,v in data.coarse_targets}
data.coarse_class_to_idx = {v:k for k,v in data.coarse_targets}
data.coarse_class_to_idx = {v:k for k,v in enumerate(data.coarse_targets)}
data
data.coarse_class_to_idx
data = CIFAR100("/fsx/pteterwak/data/positional_vpt_data")
data.coarse_targets = sparse2coarse(data.targets)
data.coarse_classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
data.coarse_class_to_idx = {v:k for k,v in enumerate(data.coarse_targets)}
data.coarse_class_to_idx
data.coarse_class_to_idx = {v:k for k,v in enumerate(data.coarse_classes)}
data.coarse_class_to_idx = {v:k for k,v in enumerate(data.coarse_classes.items())}
data.coarse_class_to_idx = {v:k for k,v in enumerate(data.coarse_classes)}
data.coarse_class_to_idx = {v:tuple(k) for k,v in enumerate(data.coarse_classes)}
data.coarse_class_to_idx = {v:tuple(k) for k,v in enumerate(data.coarse_classes)}
data.coarse_classes
data.coarse_class_to_idx = {tuple(v):k for k,v in enumerate(data.coarse_classes)}
class_inds = [torch.where(dataset.targets == class_idx)[0]
              for class_idx in dataset.class_to_idx.values()]
class_inds = [torch.where(data.targets == class_idx)[0]
              for class_idx in data.class_to_idx.values()]
class_inds = [np.where(data.targets == class_idx)[0]
              for class_idx in data.class_to_idx.values()]
class_inds
data.class_to_idx.values()
data.coarse_class_to_idx.values()
class_inds = [np.where(data.coarse_targets == class_idx)[0]
              for class_idx in data.coarse_class_to_idx.values()]
class_inds
dataloaders = [
    DataLoader(
        dataset=Subset(dataset, inds),
        batch_size=8,
        shuffle=True,
        drop_last=False)
    for inds in class_inds]
from torch.utils.data import DataLoader, Subset
dataloaders = [
    DataLoader(
        dataset=Subset(dataset, inds),
        batch_size=8,
        shuffle=True,
        drop_last=False)
    for inds in class_inds]
dataloaders = [
    DataLoader(
        dataset=Subset(data, inds),
        batch_size=8,
        shuffle=True,
        drop_last=False)
    for inds in class_inds]
dataloaders 
