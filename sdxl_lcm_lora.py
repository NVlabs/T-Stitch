# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under CC-BY-NC-SA-4.0.
# To view a copy of this license, visit 
# https://github.com/NVlabs/T-Stitch/blob/main/LICENSE.txt

from diffusers import LCMScheduler
import torch
from tstitch_sd_utils import get_tstitch_pipepline_sdxl_lcm_lora
import numpy as np
import time
import os
import torch


if __name__ == '__main__':

    large_sd = "stabilityai/stable-diffusion-xl-base-1.0"
    small_sd = "segmind/SSD-1B"
    pipe_sd = get_tstitch_pipepline_sdxl_lcm_lora(large_sd, small_sd)

    pipe_sd.scheduler = LCMScheduler.from_config(pipe_sd.scheduler.config)
    pipe_sd.to("cuda")

    prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
    neg_prompt = ""

    step_candidate = [2, 4, 8]
    for n_steps in step_candidate:

        ratios = np.arange(0, 1.1, 1/n_steps)
        latents = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.float16)

        iter_dir = f'./figures/sdxl_lcm/steps-{n_steps}'
        if not os.path.exists(iter_dir):
            os.makedirs(iter_dir)

        for ratio in ratios:
            tic1 = time.time()
            image = pipe_sd(prompt, unet_s_ratio=ratio, num_inference_steps=n_steps, neg_prompt=neg_prompt, guidance_scale=0).images[0]
            tic2 = time.time()
            print(f'ratio = {ratio}, time cost = {round((tic2 - tic1) * 1000, 0)}ms')
            image.save(f"{iter_dir}/sample-ratio-{ratio}.png")


