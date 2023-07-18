"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from .base_model import BaseModel
from .Qformer import BertConfig, BertLMHeadModel


class Blip2Base(BaseModel):

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2, num_layers=12):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.num_hidden_layers = num_layers
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def get_optimizer_params(cls, model,weight_decay, lr_scale=1):
            parameter_group_names = {}
            parameter_group_vars = {}

            for name, param in model.named_parameters():
                layer_id = None
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias"):
                    group_name = "no_decay"
                    this_weight_decay = 0.
                else:
                    group_name = "decay"
                    this_weight_decay = weight_decay
                if 'visual_encoder' in name:
                    layer_id = model.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))
                    group_name = "vit_layer_%d_%s" % (layer_id, group_name)

                if group_name not in parameter_group_names:
                    if layer_id is not None:
                        scale = lr_scales[layer_id]
                    else:
                        scale = 1
                    parameter_group_names[group_name] = {
                        "weight_decay": this_weight_decay,
                        "params": [],
                        "lr_scale": scale
                    }
                    parameter_group_vars[group_name] = {
                        "weight_decay": this_weight_decay,
                        "params": [],
                        "lr_scale": scale
                    }
                parameter_group_vars[group_name]["params"].append(param)
                parameter_group_names[group_name]["params"].append(name)
            # import json
            # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
            optim_params = list(parameter_group_vars.values())
            return optim_params

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)



