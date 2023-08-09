import copy
import datetime
import random
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.datasets import INaturalist

from util.tool import topk

from time import sleep


HOLDOUT_CLASS_IDX =  [
 [4589, 1620, 9354, 1862, 6677, 2445, 1690, 2580, 9232, 9651],
 [2927, 574, 2536, 4882, 2039, 8516, 2930, 9526, 3989, 7979],
 [2942, 3069, 8696, 7661, 7719, 2854, 2364, 8949, 4530, 8781],
 [9236, 3493, 4548, 940, 7251, 9927, 1680, 7136, 3150, 6839],
 [1791, 4631, 542, 3415, 795, 1519, 763, 5122, 9675, 4694],
 [8534, 3140, 6348, 3387, 2851, 7431, 2985, 7850, 1555, 9966],
 [1015, 2630, 524, 9381, 1290, 1340, 753, 372, 7121, 7080],
 [8566, 224, 5031, 1110, 811, 2186, 5223, 9054, 5477, 1177],
 [9276, 7470, 1232, 1394, 5463, 1890, 2501, 2621, 7855, 6749],
 [3973, 6481, 6964, 5475, 5202, 1658, 1673, 2596, 1573, 6912]] 


HOLDOUT_CLASS_IDX_LONG = np.load('holdout_class_lists/100_100_classes.npy').tolist()


class ClassSampledINaturalist(torch.utils.data.Dataset):
  def __init__(self, ds):
    super(ClassSampledINaturalist,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(len(self.ds.all_categories))]

    for i, (label, image) in enumerate(self.ds.index):
      self.class_indices[label].append(i)


  def classes(self):
    return self.class_indices

  def __getitem__(self, index):
    return self.ds[index]

  def __len__(self):
      return len(self.ds)

class ClassHoldoutINaturalist(torch.utils.data.Dataset):
  def __init__(self, ds, class_idx):
    super(ClassHoldoutINaturalist,self).__init__()
    self.ds = ds

    self.class_indices = [[] for _ in range(len(self.ds.all_categories))]

    for i, (label, image) in enumerate(self.ds.index):
      if label not in class_idx:
        self.class_indices[label].append(i)

    self.class_indices = [c for c in self.class_indices if c != []]


  def classes(self):
    return self.class_indices

  def __getitem__(self, index):
    return self.ds[index]

  def __len__(self):
      return sum([len(c) for c in self.class_indices])


class ClassSubsampledINaturalist(torch.utils.data.Dataset):
    def __init__(self, ds, class_idx):
        super(ClassSubsampledINaturalist,self).__init__()
        self.ds = ds
        self.class_idx = class_idx

        self.idx = []

        for i, (label, image) in enumerate(self.ds.index):
            if label in self.class_idx:
                self.idx.append(i)

    def __getitem__(self, index):
       return self.ds[self.idx[index]]

    def __len__(self):
         return len(self.idx)




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

def get_inat_data(data_path, batch_size, num_classes_per_batch, preprocess, preprocess_test, num_workers, holdout_class_idx, holdout_train=False, holdout_long=False):
    if holdout_long:
        class_idx = HOLDOUT_CLASS_IDX_LONG[holdout_class_idx]
    else:
        class_idx = HOLDOUT_CLASS_IDX[holdout_class_idx]
    if holdout_train:
        train_ds = ClassHoldoutINaturalist(INaturalist(data_path, version='2021_train', transform=preprocess), class_idx=class_idx)	
    else:
        train_ds = ClassSampledINaturalist(INaturalist(data_path, version='2021_train', transform=preprocess))	
    test_ds = ClassSubsampledINaturalist(INaturalist(data_path, version='2021_valid', transform=preprocess_test), class_idx=class_idx)	
    train_sampler = ClassSampler(train_ds.classes(), num_classes_per_batch ,batch_size)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_sampler = train_sampler, pin_memory=True, num_workers=num_workers)
    #train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=False, num_workers=num_workers)
    return (train_dl, test_dl)



def train_with_prompt_inaturalist(
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
             if idx > 200: 
                1/0
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
             loss.backward()

             # update the perturbation
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


def eval_with_inaturalist(
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
                 sampled_labels, remapped_labels = torch.unique(torch.Tensor(test_loader.dataset.class_idx).long(), sorted = False, return_inverse=True)
                 remap_dict = {a.item():b.item() for a,b in zip(sampled_labels, remapped_labels)}
                 remap_tuple_list= list(remap_dict.items())
                 remap_tuple_list.sort(key= lambda x: x[1])  
                 sorted_sample_labels = [a for a,b in remap_tuple_list]
                 sampled_text_inputs = text_inputs[sorted_sample_labels]
                 new_labels = torch.tensor([remap_dict[x.item()] for x in labels])
                 selected_labels = new_labels 
                 if not args.mapped and not args.qformer and not args.linear:
                     noise = prompt.perturbation.to(device)

                     images = normalization(images + noise)
                 probs = prompt(images, sampled_text_inputs, selected_labels)
                 if args.non_CLIP:
                     probs = probs[:, matching_index]
                 top1, top5 = topk(probs, (new_labels).to(device), ks=(1, 5))
                 all_top1.extend(top1.cpu())
                 all_top5.extend(top5.cpu())
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))
    print(f"top1 {np.mean(all_top1):.2%}, " f"top5 {np.mean(all_top5):.2%}")
    prompt.fixed_text_features = args.fixed_text_features
    return np.mean(all_top1), np.mean(all_top5)



def model_prompts_inaturalist(
    args,
    train_loader,
    prompt,
    text_inputs,
    pad_dim,
    normalization,
    device,
    matching_index,
):
    start_time = time.time()
    all_prompt = []
    idx = 0

    with tqdm(train_loader, total = len(train_loader.dataset)//args.batch_size , unit="batch") as tepoch: 
        for (images, labels) in  tepoch:
             # Pad the imag
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
             pred_prompt = prompt(images, sampled_text_inputs, selected_labels, return_prompt=True)

             all_prompt.append(pred_prompt)
             idx += 1

             if idx > 100:
                 break


    all_prompt_np = torch.stack(all_prompt).cpu().detach().numpy()
    mean = np.mean(all_prompt_np, axis=0)
    var = np.var(all_prompt_np, axis=0)
    1/0


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    return mean, var


