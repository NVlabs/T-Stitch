# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under CC-BY-NC-SA-4.0.
# To view a copy of this license, visit 
# https://github.com/NVlabs/T-Stitch/blob/main/LICENSE.txt

from diffusers import ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image
from tstitch_sd_utils import get_tstitch_pipepline_sdxl_controlnet
import os
import time
from controlnet_aux import OpenposeDetector



if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    cache_dir = './pretrained_models'
    save_dir = f'./figures/controlnet_pose'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    prompt = "Ironman dancing in a futuristic city, high quality"
    negative_prompt = "low quality, bad quality"

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png")
    openpose_image = openpose(image)
    openpose_image.save(f"{save_dir}/pose_image.png")

    controlnet_conditioning_scale = 0.5 
    controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16,
                                                 cache_dir=cache_dir)

    large_sd = "stabilityai/stable-diffusion-xl-base-1.0"
    small_sd = "segmind/SSD-1B"
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, cache_dir=cache_dir)

    pipe = get_tstitch_pipepline_sdxl_controlnet(large_sd, small_sd, controlnet=controlnet, vae=vae)

    ratios = np.arange(0, 1.1, 0.2)
    ratios = [round(item, 1) for item in ratios]
    latents = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.float16)

    for ratio in ratios:
        tic1 = time.time()
        image = pipe(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale,
                        image=openpose_image.resize((1024, 1024)), unet_s_ratio=ratio,
                        negative_prompt=negative_prompt).images[0]
        tic2 = time.time()
        print(f'ratio = {ratio}, time cost = {round(tic2 - tic1, 1)}s')
        image.save(f"{save_dir}/sample-ratio-{ratio}.png")

    with open(f'{save_dir}/prompt.txt', 'w') as f:
        f.write(prompt + '\n')
        f.write(negative_prompt + '\n')