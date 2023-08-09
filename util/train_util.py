import copy
import datetime
import random
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F

from util.tool import topk

from time import sleep


class ClassSampler():
    def __init__(self, classes, num_classes_per_batch, batch_size):
        self.classes = classes
        self.grouped = None
        self.num_classes_per_batch = num_classes_per_batch
        self.batch_size = batch_size
        self.class_idx = list(range(len(self.classes)))

    def __iter__(self):
        classes = copy.deepcopy(self.classes)
        class_idx = copy.deepcopy(self.class_idx)
        class_choice = random.sample(class_idx, k = self.num_classes_per_batch)
        global_res = []
        res = []
        popped = 0
        to_pop = []
        total_samples = sum([len(c) for c in classes])
        #print(total_samples)
        #sleep(10)
        #pbar = tqdm(total=total_samples)
        while len(class_idx) > 0:
          #print("another looop")
          num_classes = min(self.num_classes_per_batch - popped , len(class_idx))

          for i in class_choice:
            #print("class")
            #print(class_choice)
            res.append(classes[i].pop())
            if len(classes[i]) == 0:
              to_pop.append(i)
              popped += 1
            if len(res) == self.batch_size:
              #print("sample")
              #print(res)
              #global_res.append(res)
              yield res
              #pbar.update(len(res))
              res = []
              popped = 0
              if len(to_pop) > 0:
                class_choice = [e for  e in class_choice if e not in to_pop]
                class_idx = [e for  e in class_idx if e not in to_pop]
                to_pop = []
              num_classes = min(self.num_classes_per_batch - popped , len(class_idx))
              class_choice = random.sample(class_idx, k = num_classes)
              break
          if len(to_pop) > 0:
              class_choice = [e for  e in class_choice if e not in to_pop]
              class_idx = [e for  e in class_idx if e not in to_pop]
              to_pop = []
          num_classes = min(self.num_classes_per_batch - popped , len(class_idx))
          if num_classes == 0 and len(res) > 0:
              #global_res.append(res)
              yield res
              res = []
              popped = 0
              num_classes = min(self.num_classes_per_batch - popped , len(class_idx))
              class_choice = random.sample(class_idx, k = num_classes)

              #pbar.update(len(res))
        #pbar.close()
        #return global_res


def train_with_prompt_vpn(
    args,
    epoch,
    train_loader,
    prompt,
    text_inputs,
    pad_dim,
    criterion,
    optim,
    text_optim,
    normalization,
    device,
    matching_index,
    schedule
):
    start_time = time.time()
    lr = optim.param_groups[0]["lr"]
    all_loss = []
    all_top1 = []
    idx = 0

    with tqdm(train_loader, total = len(train_loader.dataset)//args.batch_size , unit="batch") as tepoch: 
        for (images, labels) in  tepoch:
             # Pad the imag
             if args.qformer or args.linear:
                schedule.step(epoch, idx)
             tepoch.set_description(f"Epoch {epoch}")
             images = F.pad(images, pad_dim, "constant", value=0)
             images = images.to(device)
             labels = labels.to(device)
             sampled_labels, remapped_labels = torch.unique(labels , sorted=False, return_inverse=True)
             if not args.fixed_text_features:
                sampled_text_inputs = text_inputs[sampled_labels]
             else: 
                 sampled_text_inputs = text_inputs
                 repmapped_labels = labels
             selected_labels = sampled_labels 
             if not args.mapped and not args.qformer and not args.linear:
                 noise = prompt.perturbation.to(device)
                 noise = noise.repeat(images.size(0), 1, 1, 1)
                 noise.retain_grad()

                 # Normalize the image and noise
                 images = normalization(images + noise)
                 images.require_grad = True
             probs = prompt(images, sampled_text_inputs, selected_labels)
             if args.non_CLIP:
                 probs = probs[:, matching_index]
             loss = criterion(probs, remapped_labels)
             if args.mapped or args.qformer or args.linear:
                 optim.zero_grad()
             if args.prompt_tuning:
                text_optim.zero_grad()
             loss.backward()

             # update the perturbation
             if args.prompt_tuning:
                text_optim.step()
             if not args.disable_visual_prompt_tuning:
                if args.mapped or args.qformer or args.linear:
                    optim.step()
                else:
                    grad_p_t = noise.grad
                    grad_p_t = grad_p_t.mean(0).squeeze(0)
                    g_norm = torch.norm(grad_p_t.view(-1), dim=0).view(1, 1, 1)
                    scaled_g = grad_p_t / (g_norm + 1e-10)
                    scaled_g_pad = scaled_g * prompt.mask.to(device)
                    updated_pad = scaled_g_pad * lr
                    prompt.perturbation.data = prompt.perturbation.data - updated_pad.detach().cpu()
                    prompt.zero_grad()

             all_loss.append(loss.detach().cpu().numpy())
             top1 = topk(probs, (remapped_labels).to(device), ks=(1,))[0]
             all_top1.extend(top1.cpu())
             idx += 1
             if (idx % 100 == 0):
                tepoch.set_postfix(loss = np.mean(all_loss), accuracy = np.mean(all_top1))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(
        "At the {} epoch, the Lr is {}, the top1 is {} and training time  is {}".format(
            str(epoch), str(lr), str(
                np.mean(all_top1)), total_time_str))

    return np.mean(all_loss), np.mean(all_top1)

def eval_with_vpn(
        args,
        test_loader,
        prompt,
        pad_dim,
        text_inputs,
        normalization,
        device,
        matching_index, 
        class_idx_list):
    start_time = time.time()
    all_top1, all_top5 = [], []
    prompt.fixed_text_features = False
    label_map =  {v:k for k,v in enumerate(class_idx_list)}
    print("starting evaluation")
    with tqdm(test_loader, total = len(test_loader.dataset)//args.batch_size , unit="batch") as teval: 
        for images, labels in teval:
             with torch.no_grad():
                 images = F.pad(images, pad_dim, "constant", value=0)
                 images = images.to(device)
                 labels = torch.tensor([label_map[x.item()] for x in labels])
                 labels = labels.to(device)
                 sampled_text_inputs = text_inputs
                 selected_labels = class_idx_list
                 if not args.mapped and not args.qformer and not args.linear:
                     noise = prompt.perturbation.to(device)

                     images = normalization(images + noise)
                 probs = prompt(images, sampled_text_inputs, selected_labels)
                 if args.non_CLIP:
                     probs = probs[:, matching_index]
                 (top1,) = topk(probs, (labels).to(device), ks=(1,))
                 all_top1.extend(top1.cpu())
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))
    print(f"top1 {np.mean(all_top1):.2%}")
    prompt.fixed_text_features = args.fixed_text_features
    return np.mean(all_top1), np.mean(all_top5)




