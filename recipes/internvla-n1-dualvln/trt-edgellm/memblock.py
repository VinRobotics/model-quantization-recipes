# SPDX-FileCopyrightText: Copyright (c) 2026 VinRobotics
# SPDX-License-Identifier: BSD-3-Clause
"""System 1 memory-encoding block, wrapped as a single exportable module.

Mirrors the reference `generate_traj` (nextdit_async) memory path:
    rgb_model.get_intermediate_layers -> memory_encoder -> concat -> rgb_resampler
Input : images_dp_norm  [T, 3, 224, 224]   (T frames, ImageNet-normalized)
Output: memory_tokens   [1, 32, 768]
"""
import torch


class MemBlock(torch.nn.Module):
    def __init__(self, rgb_model, memory_encoder, rgb_resampler):
        super().__init__()
        self.rgb = rgb_model
        self.me = memory_encoder
        self.rr = rgb_resampler

    def forward(self, x):  # x = [T, 3, 224, 224]
        feat = self.rgb.get_intermediate_layers(x)[0].unflatten(dim=0, sizes=(1, -1))  # [1, T, Np, 384]
        f = feat.flatten(1, 2)                                                          # [1, T*Np, 384]
        mf = self.me(f)
        mf = torch.cat([f, mf], dim=-1)
        return self.rr(mf)
