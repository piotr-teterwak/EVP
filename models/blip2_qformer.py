"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from .blip2 import Blip2Base, LayerNorm


class Blip2Qformer(Blip2Base):
    """
    """

    def __init__(
        self,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        input_width = 2048,
        num_layers = 12
    ):
        super().__init__()


        self.ln = LayerNorm(input_width)
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, input_width, cross_attention_freq,
            num_layers
        )
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.proj.weight.data.fill_(0.00)
        self.proj.bias.data.fill_(0.00)
        



    def forward(self, input_sequence):
        embeds = self.ln(input_sequence)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(
            input_sequence.device
        )

        query_tokens = self.query_tokens.expand(embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=embeds,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        return self.proj(query_output.last_hidden_state)


