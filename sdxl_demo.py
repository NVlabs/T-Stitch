# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under CC-BY-NC-SA-4.0.
# To view a copy of this license, visit 
# https://github.com/NVlabs/T-Stitch/blob/main/LICENSE.txt

import numpy as np
import torch
from tstitch_sd_utils import get_tstitch_pipepline_sdxl
from torch import Generator
import torch
import time
import os

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    save_dir = f'./figures/sdxl_demo'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    large_sd = "stabilityai/stable-diffusion-xl-base-1.0"
    small_sd = "segmind/SSD-1B"
    pipe_sd = get_tstitch_pipepline_sdxl(large_sd, small_sd)


    prompt = "concept art of dragon flying over town, clouds. digital artwork, illustrative, painterly, matte painting, highly detailed, cinematic composition"
    negative_prompt = "photo, photorealistic, realism, ugly"

    ratios = np.arange(0, 1.1, 0.1)
    ratios = [round(item, 1) for item in ratios]
    latents = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.float16)

    for ratio in ratios:
        tic1 = time.time()
        image = pipe_sd(prompt, unet_s_ratio=ratio, latents=latents, negative_prompt=negative_prompt).images[0]
        tic2 = time.time()
        print(f'ratio = {ratio}, time cost = {round(tic2 - tic1, 1)}s')
        image.save(f"{save_dir}/sample-ratio-{ratio}.png")

    with open(f'{save_dir}/prompt.txt', 'w') as f:
        f.write(prompt)