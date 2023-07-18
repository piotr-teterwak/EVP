import os
import random
import time
import clip
import torch
import torchvision
import wandb
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import datetime
import torch.nn.functional as F
import argparse
import torchvision.models as models
from models.blip2_qformer import Blip2Qformer
from util.optims import LinearWarmupCosineLRScheduler
from util.tool import refine_classname, topk, _convert_image_to_rgb, add_weight_decay
from util.get_index import get_index
from util.data import cifar_100_coarse_labels, sparse2coarse
from util.inaturalist_fns import get_inat_data, train_with_prompt_inaturalist, eval_with_inaturalist
from torchvision.transforms import (
    Compose,
    ToTensor,
    InterpolationMode,
)

_tokenizer = _Tokenizer()

SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
SLURM_ARRAY_TASK_ID = os.environ.get('SLURM_ARRAY_TASK_ID')

class Pertubation(torch.nn.Module):
    def __init__(self, pad_h, pad_w, clip_model, mapped=True, normalization=None, bias=False, qformer=False, num_layers=12, representation_path=None, num_dummy_classes = 0, fixed_text_features = None, linear_probe_classes = -1):
        super().__init__()
        mask = torch.ones((3, 224, 224))
        mask[:, pad_h: 224 - pad_h, pad_w: 224 - pad_w] = 0
        self.register_buffer("mask", mask)
        self.mapped = mapped 
        self.normalization = normalization
        self.bias = bias
        self.qformer = qformer
        self.fixed_text_features = fixed_text_features
        self.encoded_features = None
        self.linear_probe_classes = linear_probe_classes

        if self.mapped:
            self.perturbation = None
            self.register_buffer("random_tensor_input",torch.rand(1,784))
            self.prompt_map_fn = torch.nn.Linear(784,3*224*224, bias=self.bias)
            self.prompt_map_fn.weight.data.fill_(0.00)
            if self.bias:
                self.prompt_map_fn.bias.data.fill_(0.00)
        elif self.qformer:
            self.perturbation = None
            if num_dummy_classes > 0:
                self.representation_array = torch.nn.ParameterList([torch.nn.Parameter(torch.ones((10,6656)), requires_grad=False) for idx in range(num_dummy_classes)])
            else:
                class_representations = np.load(representation_path)
                self.representation_array = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(class_representations[idx]), requires_grad=False) for idx in class_representations])
            self.qformer = Blip2Qformer(input_width = 6656, embed_dim= 3*224*224, num_layers=num_layers)
        else:
            delta = torch.zeros((3, 224, 224))
            delta.require_grad = True
            self.perturbation = torch.nn.Parameter(
            delta.float(), requires_grad=True)

        if self.linear_probe_classes > -1:
            self.linear = torch.nn.Embedding(self.linear_probe_classes, clip_model.visual.output_dim).half()
        self.model = clip_model

    def set_text_features(self, input_text):
        if self.fixed_text_features:
            self.encoded_features = self.model.encode_text(input_text)
        if self.linear_probe_classes > -1:
            text_features = self.model.encode_text(input_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
            self.linear.weight.data.copy_(text_features)

    def forward(self, images, text_inputs, selected_labels=None):
        if self.mapped:
            prompt_map = self.prompt_map_fn(self.random_tensor_input)
            self.perturbation = prompt_map.view(3,224,224)
            noise = self.perturbation.repeat(images.size(0),1,1,1)
            images = self.normalization(noise + images)
        elif self.qformer:
            if selected_labels is not None:
                reps = torch.concat([self.representation_array[s] for s in selected_labels]).view(1,-1,6656)
            else:
                reps = torch.concat(list(self.representation_array)).view(1,-1,6656)
            prompts = self.qformer(reps)[0,-1,:]
            self.perturbation = prompts.view(3,224,224)
            noise = self.perturbation * self.mask 
            noise = noise.repeat(images.size(0),1,1,1)
            images = self.normalization(noise + images)
 
        image_features = self.model.encode_image(images)
        if self.linear_probe_classes > -1:
            selected_labels = selected_labels.to(self.linear.weight.device)
            norm_text_features = self.linear(selected_labels)
        elif not self.fixed_text_features:
            text_features = self.model.encode_text(text_inputs)
        else:
            text_features = self.encoded_features
        norm_image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        if self.linear_probe_classes < 0:
            norm_text_features = text_features / \
                text_features.norm(dim=-1, keepdim=True)

        probs = (
            self.model.logit_scale.exp()
            * norm_image_features
            @ norm_text_features.T
        )

        return probs


class Pertubation_non_CLIP(torch.nn.Module):
    def __init__(self, pad_h, pad_w, model):
        super().__init__()
        self.mask = torch.ones((3, 224, 224))
        self.mask[:, pad_h: 224 - pad_h, pad_w: 224 - pad_w] = 0

        delta = torch.zeros((3, 224, 224))

        delta.require_grad = True
        self.perturbation = torch.nn.Parameter(
            delta.float(), requires_grad=True)

        self.model = model

    def forward(self, images, text_inputs=None):
        probs = self.model(images)

        return probs


def parse_option():
    parser = argparse.ArgumentParser("Visual Prompting for CLIP")


    parser.add_argument(
        '--map-devices',
        action='store_true',
        help='map devices')



    # training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="batch_size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="num of workers to use"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of training epoch5s"
    )
    parser.add_argument(
        "--linear-probe-classes", type=int, default=-1, help="number of lp classes"
    )

    parser.add_argument(
        '--mapped',
        action='store_true',
        help='mapped_fn')
    parser.add_argument(
        '--qformer',
        action='store_true',
        help='use qformer')
    parser.add_argument(
        '--constant-conditioning',
        action='store_true',
        help='use qformer')
    parser.add_argument(
        '--fixed_text_features',
        action='store_true',
        help='precompute text features')
    parser.add_argument(
        "--num_qformer_layers", type=int, default=12, help="number of qformer layers"
    )




    # optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=70,
        help="learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="weight decay")


    # model
    parser.add_argument("--arch", type=str, default="ViT-B/32")
    parser.add_argument(
        '--non_CLIP',
        action='store_true',
        help='Perform evaluation only')
    parser.add_argument(
        '--bias',
        action='store_true',
        help='Perform evaluation only')


    parser.add_argument(
        '--non_CLIP_model',
        type=str,
        default='rn50',
        choices=[
            'rn50',
            'instagram_resnext101_32x8d'],
        help='The non CLIP Model')
    parser.add_argument(
        '--index_path',
        type=str,
        default='argmax_rn50_cifar10.pickle',
        help='The path of matched index')
    parser.add_argument(
        "--prompt_size", type=int, default=30, help="size for visual prompts"
    )

    # dataset
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.expanduser("~/.cache"),
        help="dataset")
    parser.add_argument(
        "--representation_path",
        type=str,
        default='representations.npz',
        help="representation_path")
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR100",
        help="dataset")
    parser.add_argument(
        "--image_size",
        type=int,
        default=164,
        help="image size")
    parser.add_argument(
        '--sample_coarse_labels',
        action='store_true',
        help='Do coarse label sampling for zero shot')
    parser.add_argument(
        '--test_coarse_label',
        type=int,
        default=0,
        help='Which coarse label for eval')
    parser.add_argument(
        '--num_classes_per_batch',
        type=int,
        default=10,
        help='Num Classes per batch')
    parser.add_argument(
        '--num_dummy_classes',
        type=int,
        default=0,
        help='Number of dummy classes')
    parser.add_argument(
        "--holdout_class_idx", type=int, default=9, help="eval class list"
    )
    parser.add_argument(
        '--holdout_train',
        action='store_true',
        help='Holdout training samples')







    # save
    parser.add_argument(
        "--save_path",
        type=str,
        default="./save/models",
        help="path to save models")

    # seed
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for initializing training"
    )

    # eval
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Perform evaluation only')

    parser.add_argument(
        "--checkpoint", type=str, help="The checkpoint of trained model"
    )

    # wandb
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="whether to use wandb")
    parser.add_argument(
        "--project",
        type=str,
        default="visual prompting",
        help="The name of wandb project name",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="cifar100",
        help="The name of wandb job name")
    parser.add_argument(
        "--entity", type=str, default="", help="Your user name of wandb"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_option()

    if args.map_devices:
        device_map = args.test_coarse_label % 2
        if args.constant_conditioning:
            device_map +=2
        device = "cuda:{}".format(device_map) if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # log setting
    log_wandb = args.use_wandb
    project = args.project
    job_name = args.job_name
    save_path = args.save_path
    if SLURM_JOB_ID is not None:
       save_path = os.path.join(save_path, SLURM_JOB_ID)
    if SLURM_ARRAY_TASK_ID is not None:
        save_path = os.path.join(save_path, SLURM_ARRAY_TASK_ID)


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if log_wandb:
        wandb.init(
            project=str(project),
            name=str(job_name),
            entity=args.entity)

    # Load the clip model
    clip_model, preprocess = clip.load(args.arch, device)
    _, preprocess_test = clip.load(args.arch, device)

    # Load the non clip model
    if args.non_CLIP:
        print('use model:' , args.non_CLIP_model)
        _model = args.non_CLIP_model

        if _model == 'rn50':
            model = models.__dict__['resnet50'](pretrained=True).to(device)

        elif _model == 'instagram_resnext101_32x8d':
            model = torch.hub.load(
                'facebookresearch/WSL-Images',
                'resnext101_32x8d_wsl').to(device)

    # Prepare the dataset
    # Normalize the image and noise together
    normalization = preprocess.transforms[-1]
    preprocess_test.transforms.pop(-1)
    preprocess = Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(
                args.image_size, interpolation=InterpolationMode.BICUBIC
            ),
            torchvision.transforms.RandomCrop(args.image_size),
            _convert_image_to_rgb,
            ToTensor(),
        ]
    )
    preprocess_test = Compose(
        [
            torchvision.transforms.Resize(
                args.image_size,
                interpolation=InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(
                size=(
                    args.image_size,
                    args.image_size)),
            _convert_image_to_rgb,
            ToTensor(),
        ])


    if args.dataset == 'inaturalist':
        train_loader, test_loader = get_inat_data(args.root, args.batch_size, args.num_classes_per_batch, preprocess, preprocess_test, args.num_workers, args.holdout_class_idx, args.holdout_train)
        classes = train_loader.dataset.ds.all_categories
        classes = [' '.join(c.split('_')) for c in classes]
        text_inputs = torch.concat([clip.tokenize(f"this is a photo of a {c}").to(device) for c in classes])
        train_text_inputs = text_inputs
        test_text_inputs = text_inputs
    elif args.sample_coarse_labels:
        train_set = CIFAR100(
             args.root,
             download=True,
             train=True,
             transform=preprocess)
        test_set = CIFAR100(
             args.root,
             download=True,
             train=False,
             transform=preprocess_test)
     
        train_set.coarse_targets = sparse2coarse(train_set.targets)
        test_set.coarse_targets = sparse2coarse(test_set.targets)
        train_set.coarse_classes = cifar_100_coarse_labels[:args.test_coarse_label] + cifar_100_coarse_labels[args.test_coarse_label + 1:]
        test_set.coarse_classes = [cifar_100_coarse_labels[args.test_coarse_label]]
        train_set.coarse_class_to_idx = {tuple(v):k for k,v in enumerate(cifar_100_coarse_labels)}
        test_set.coarse_class_to_idx = {tuple(v):k for k,v in enumerate(cifar_100_coarse_labels)}
        train_class_inds = [np.where(train_set.coarse_targets == class_idx)[0] 
                for class_idx in train_set.coarse_class_to_idx.values()]
        test_class_inds = [np.where(test_set.coarse_targets == class_idx)[0] 
                for class_idx in test_set.coarse_class_to_idx.values()]

        train_dataloaders = [
            DataLoader(
                dataset=Subset(train_set, inds),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                pin_memory=True,
                num_workers=args.num_workers)
            for inds in train_class_inds]
        train_loader = train_dataloaders[:args.test_coarse_label] + train_dataloaders[args.test_coarse_label+1:]

        test_loader = DataLoader(
                dataset=Subset(test_set, test_class_inds[args.test_coarse_label]
),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False)
        
        text_inputs = [torch.cat([clip.tokenize(f"this is a photo of a {c}") for c in classes_names]).to(device) for classes_names in cifar_100_coarse_labels]

        for t in text_inputs:
            t.requires_grad = False

        test_text_inputs = text_inputs[args.test_coarse_label]
        train_text_inputs = text_inputs[:args.test_coarse_label] + text_inputs[args.test_coarse_label + 1:]
 
    else:

         train_set = CIFAR100(
             args.root,
             download=True,
             train=True,
             transform=preprocess)
         test_set = CIFAR100(
             args.root,
             download=True,
             train=False,
             transform=preprocess_test)
         
         classes_names = train_set.classes
         classes_names = refine_classname(classes_names)
         text_inputs = torch.cat(
             [clip.tokenize(f"this is a photo of a {c}") for c in classes_names]
         )
         text_inputs.requires_grad = False  
         text_inputs = text_inputs.to(device)
         train_text_inputs = text_inputs
         test_text_inputs = text_inputs

         train_loader = DataLoader(
             train_set,
             batch_size=args.batch_size,
             shuffle=True,
             pin_memory=True,
             num_workers=args.num_workers,
         )
         test_loader = DataLoader(
             test_set,
             batch_size=args.batch_size,
             shuffle=False,
             pin_memory=True,
             num_workers=args.num_workers,
         )




    matching_index = []
    if args.non_CLIP:
        print('start matching the index')
        matching_index = get_index(train_set, model, device)

    # Training setting
    epoch = args.epochs
    lr = args.learning_rate

    # Initialize the prompt

    if not args.non_CLIP:
        prompt = Pertubation(args.prompt_size, args.prompt_size, clip_model, args.mapped, normalization, args.bias, args.qformer, args.num_qformer_layers, args.representation_path, args.num_dummy_classes, args.fixed_text_features, args.linear_probe_classes)
        if args.mapped or args.qformer:
            prompt.to(device)
    else:
        prompt = Pertubation_non_CLIP(
            args.prompt_size, args.prompt_size, model)
    pad_length = int((224 - args.image_size) / 2)
    pad_dim = (pad_length, pad_length, pad_length, pad_length)

    # Optimizer setting
    prompt.model.requires_grad_(False)
    if args.linear_probe_classes > -1:
        prompt.linear.requires_grad_(True)
    if args.qformer:
        #fix optimizer and params 
        param_groups = prompt.qformer.get_optimizer_params(prompt, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-05)
        schedule = LinearWarmupCosineLRScheduler(optimizer, epoch, 0.0, lr, 2000,0)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        param_groups = add_weight_decay(prompt, 0.0, skip_list=("perturbation", "prompt_map_fn.weight", "prompt_map_fn.bias"))
        print(param_groups)
        optimizer = torch.optim.SGD(param_groups, lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epoch)
    if args.dataset == 'inaturalist':
        train_fn = train_with_prompt_inaturalist 
        eval_fn = eval_with_inaturalist
    elif args.sample_coarse_labels:
        train_fn = train_with_prompt_coarse_labels
        eval_fn = eval_with_coarse_labels
    else:
        train_fn = train_with_prompt
        eval_fn = eval

    max_acc = 0
    if  args.fixed_text_features or args.linear_probe_classes > -1:
        prompt.set_text_features(train_text_inputs)
    # Begin training
    if not args.evaluate:
        if log_wandb:
            wandb.watch(prompt)
        print('Start Training')
        for e in range(epoch):
           # train_loss, train_top1 = train_fn(
           #     args,
           #     epoch=e,
           #     train_loader=train_loader,
           #     prompt=prompt,
           #     text_inputs=train_text_inputs,
           #     pad_dim=pad_dim,
           #     criterion=criterion,
           #     optim=optimizer,
           #     normalization=normalization,
           #     device=device,
           #     matching_index=matching_index, 
           #     schedule = schedule
           # )
            if not args.qformer:
                schedule.step()
            test_acc1, test_acc5 = eval_fn(
                args,
                test_loader=test_loader,
                prompt=prompt,
                pad_dim=pad_dim,
                text_inputs=test_text_inputs,
                normalization=normalization,
                device=device,
                matching_index=matching_index
            )
            if test_acc1 > max_acc:
                max_acc = test_acc1
            model_state = prompt.state_dict()
            if not args.mapped and not args.qformer:
                 save_dict = {"perturbation": model_state["perturbation"]}
            else:
                save_dict = model_state
            save_path = args.save_path
            if SLURM_JOB_ID is not None:
               save_path = os.path.join(save_path, SLURM_JOB_ID)
            if SLURM_ARRAY_TASK_ID is not None:
                save_path = os.path.join(save_path, SLURM_ARRAY_TASK_ID)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            torch.save(save_dict, save_path + "/checkpoint.pth")
            print("max acc is {}".format(str(max_acc)))
            if log_wandb:
                log_stauts = {
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": train_loss,
                    "train_top1": train_top1,
                    "test_acc1": test_acc1,
                    "test_acc5": test_acc5,
                }
                wandb.log(log_stauts, step=e)

    # Begin testing
    else:
        print('Start Evaluating')
        # Load the model
        checkpoint = args.checkpoint
        state_dict = torch.load(checkpoint, map_location="cpu")
        perturbation_state = prompt.state_dict()
        perturbation_state["perturbation"] = state_dict["perturbation"]
        prompt.load_state_dict(perturbation_state)

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_acc1, test_acc5 = eval_fn(
            test_loader=test_loader,
            prompt=prompt,
            pad_dim=pad_dim,
            text_inputs=text_inputs,
            normalization=normalization,
            device=device,
            matching_index=matching_index
        )
        print("Test acc1 is {}".format(str(test_acc1)))


def train_with_prompt(
    args,
    epoch,
    train_loader,
    prompt,
    text_inputs,
    pad_dim,
    criterion,
    optim,
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

    for i, (images, labels) in enumerate(tqdm(train_loader)):
        # Pad the image
        if args.qformer:
            schedule.step(epoch, i)
        images = F.pad(images, pad_dim, "constant", value=0)
        images = images.to(device)
        if not args.mapped and not args.qformer:
            noise = prompt.perturbation.to(device)
            noise = noise.repeat(images.size(0), 1, 1, 1)
            noise.retain_grad()

            # Normalize the image and noise
            images = normalization(images + noise)
            images.require_grad = True

        probs = prompt(images, text_inputs)
        if args.non_CLIP:
            probs = probs[:, matching_index]
        loss = criterion(probs, (labels).to(device))
        if args.mapped or args.qformer:
            optim.zero_grad()
        loss.backward()

        # update the perturbation
        if args.mapped or args.qformer:
            optim.step()
            pass
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
        top1, top5 = topk(probs, (labels).to(device), ks=(1, 5))
        all_top1.extend(top1.cpu())
        idx += 1

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(
        "At the {} epoch, the Lr is {}, the top1 is {} and training time  is {}".format(
            str(epoch), str(lr), str(
                np.mean(all_top1)), total_time_str))

    return np.mean(all_loss), np.mean(all_top1)


def train_with_prompt_coarse_labels(
    args,
    epoch,
    train_loader,
    prompt,
    text_inputs,
    pad_dim,
    criterion,
    optim,
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


    iterators = [iter(i) for i in train_loader]
    iterators = list(zip(iterators,text_inputs, range(len(iterators))))
   
    total_length = sum([len(l) for l in train_loader])
    pbar = tqdm(total=total_length)
    while iterators:
        choice_idx = random.choice(range(len(iterators)))
        iterator = iterators[choice_idx]
        try:
            images, labels = next(iterator[0])
            text_embedding = iterator[1]
            coarse_class = iterator[2]
        except StopIteration:
            iterators.remove(iterator)
            continue
        # Pad the image
        fine_labels = train_loader[0].dataset.dataset.coarse_classes[coarse_class]
        class_map = train_loader[0].dataset.dataset.class_to_idx
        label_map = np.zeros(100) 
        if not args.constant_conditioning:
            selected_labels = [class_map[l] for l in fine_labels]
        else:
            selected_labels = None
        for i,l in enumerate(fine_labels):
            label_map[class_map[l]] = i
        remapped_labels = torch.Tensor(label_map[labels]).type(torch.LongTensor)
        schedule.step(epoch, idx)
        images = F.pad(images, pad_dim, "constant", value=0)
        images = images.to(device)
        if not args.mapped and not args.qformer:
            noise = prompt.perturbation.to(device)
            noise = noise.repeat(images.size(0), 1, 1, 1)
            noise.retain_grad()

            # Normalize the image and noise
            images = normalization(images + noise)
            images.require_grad = True
        probs = prompt(images, text_embedding, selected_labels)
        if args.non_CLIP:
            probs = probs[:, matching_index]
        loss = criterion(probs, (remapped_labels).to(device))
        if args.mapped or args.qformer:
            optim.zero_grad()
        loss.backward()

        # update the perturbation
        if args.mapped or args.qformer:
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
        top1, top5 = topk(probs, (remapped_labels).to(device), ks=(1, 5))
        #print("Top 1 {}".format(np.mean(top1.cpu().numpy())))
        all_top1.extend(top1.cpu())
        idx += 1
        pbar.update(1)

    pbar.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(
        "At the {} epoch, the Lr is {}, the top1 is {} and training time  is {}".format(
            str(epoch), str(lr), str(
                np.mean(all_top1)), total_time_str))

    return np.mean(all_loss), np.mean(all_top1)




def eval(
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
    print("starting evaluation")
    for images, labels in tqdm(test_loader):
        with torch.no_grad():
            images = F.pad(images, pad_dim, "constant", value=0)
            images = images.to(device)
            if not args.mapped and not args.qformer:
                noise = prompt.perturbation.to(device)

                images = normalization(images + noise)
            probs = prompt(images, text_inputs)
            if args.non_CLIP:
                probs = probs[:, matching_index]
            top1, top5 = topk(probs, (labels).to(device), ks=(1, 5))
            all_top1.extend(top1.cpu())
            all_top5.extend(top5.cpu())
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))
    print(f"top1 {np.mean(all_top1):.2%}, " f"top5 {np.mean(all_top5):.2%}")
    return np.mean(all_top1), np.mean(all_top5)




def eval_with_coarse_labels(
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
    print("starting evaluation")
    for images, labels in tqdm(test_loader):
        with torch.no_grad():
            fine_labels = test_loader.dataset.dataset.coarse_classes[0]
            class_map = test_loader.dataset.dataset.class_to_idx
            label_map = np.zeros(100) 
            if not args.constant_conditioning:

                selected_labels = [class_map[l] for l in fine_labels]
            else:
                selected_labels = None

            for i,l in enumerate(fine_labels):
                label_map[class_map[l]] = i
            remapped_labels = torch.Tensor(label_map[labels]).type(torch.LongTensor)
        
            images = F.pad(images, pad_dim, "constant", value=0)
            images = images.to(device)
            if not args.mapped and not args.qformer:
                noise = prompt.perturbation.to(device)

                images = normalization(images + noise)
            probs = prompt(images, text_inputs, selected_labels)
            if args.non_CLIP:
                probs = probs[:, matching_index]
            top1, top5 = topk(probs, (remapped_labels).to(device), ks=(1, 5))
            all_top1.extend(top1.cpu())
            all_top5.extend(top5.cpu())
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))
    print(f"top1 {np.mean(all_top1):.2%}, " f"top5 {np.mean(all_top5):.2%}")
    return np.mean(all_top1), np.mean(all_top5)




if __name__ == "__main__":
    main()
