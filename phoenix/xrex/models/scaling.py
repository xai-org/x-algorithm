# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import math

from xai_configlib import Config, configclass

from xrex.utils.utils import ffn_size


@configclass
class ScaleConfig(Config):
    qk_norm_lr_scale: float = 1.0
    ln_weight_decay_mask: float = 0.0
    ffn_init_scale: float = 1.0
    attn_init_scale: float = 1.0

    def ln_lr_multiplier(self):
        return 1.0

    def qk_norm_lr_multiplier(self):
        return self.qk_norm_lr_scale

    def emb_lr_multiplier(self, emb_size):
        return math.sqrt(512 / emb_size) * 0.5

    def hidden_lr_multiplier(self, fan_in):
        return 512 / fan_in

    def ffn_output_lr_multiplier(self, fan_in, gated_mlp):
        return ffn_size(512, 3.0, gated_mlp=gated_mlp) / fan_in * (2.0 / 3.0)

    def attn_output_lr_multiplier(self, fan_in):
        return 128 * 4 / fan_in

    def input_scale(self, emb_size):
        return math.sqrt(emb_size)

    def output_scale(self, emb_size):
        return math.sqrt(512 / emb_size) * 2.0

    def attn_output_scale(self, key_size):
        return math.sqrt(128) / key_size
