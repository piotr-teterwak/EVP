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

def get_flowers_data(data_path, batch_size, preprocess, preprocess_test, num_workers):
    test_ds = Flowers102(root=data_path, transform = preprocess_test, split='test')
    train_dl = None
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=False, num_workers=num_workers)
    return (train_dl, test_dl)




def eval_with_flowers(
        args,
        test_loader,
        prompt,
        pad_dim,
        text_inputs,
        normalization,
        device,
        matching_index):
    start_time = time.time()
    all_top1, all_top5 = [], []
    prompt.fixed_text_features = False
    print("starting evaluation")
    with tqdm(test_loader, total = len(test_loader.dataset)//args.batch_size , unit="batch") as teval: 
        for images, labels in teval:
             with torch.no_grad():
                 images = F.pad(images, pad_dim, "constant", value=0)
                 images = images.to(device)
                 labels =  labels.to(device)
                 sampled_text_inputs = text_inputs
                 selected_labels = np.arange(102)
                 if not args.mapped and not args.qformer and not args.linear:
                     noise = prompt.perturbation.to(device)

                     images = normalization(images + noise)
                 probs = prompt(images, sampled_text_inputs, selected_labels)
                 if args.non_CLIP:
                     probs = probs[:, matching_index]
                 top1, top5 = topk(probs, (labels).to(device), ks=(1, 5))
                 all_top1.extend(top1.cpu())
                 all_top5.extend(top5.cpu())
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))
    print(f"top1 {np.mean(all_top1):.2%}, " f"top5 {np.mean(all_top5):.2%}")
    prompt.fixed_text_features = args.fixed_text_features
    return np.mean(all_top1), np.mean(all_top5)




