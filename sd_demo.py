# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under CC-BY-NC-SA-4.0.
# To view a copy of this license, visit 
# https://github.com/NVlabs/T-Stitch/blob/main/LICENSE.txt

import torch
from tstitch_sd_utils import get_tstitch_pipepline
import os

large_sd = "Envvi/Inkpunk-Diffusion"
small_sd = "nota-ai/bk-sdm-tiny"

pipe_sd = get_tstitch_pipepline(large_sd, small_sd)
prompt = 'a squirrel in the park, nvinkpunk style'
latent = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)

save_dir = f'figures/inkpunk'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

ratios = [round(item, 1) for item in torch.arange(0, 1.1, 0.1).tolist()]
for ratio in ratios:
    image = pipe_sd(prompt, unet_s_ratio=ratio, latents=latent, height=512, width=512).images[0]
    image.save(f"{save_dir}/sample-ratio-{ratio}.png")