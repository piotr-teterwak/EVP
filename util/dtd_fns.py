import copy
import datetime
import random
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.datasets import DTD

from util.tool import topk

from time import sleep

def get_dtd_data(data_path, batch_size, preprocess, preprocess_test, num_workers):
    test_ds = DTD(root=data_path, transform = preprocess_test, split='test')
    train_dl = None
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=False, num_workers=num_workers)
    return (train_dl, test_dl)




def eval_with_dtd(
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
                 selected_labels = np.arange(47)
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




