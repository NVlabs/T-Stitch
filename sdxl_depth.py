# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under CC-BY-NC-SA-4.0.
# To view a copy of this license, visit 
# https://github.com/NVlabs/T-Stitch/blob/main/LICENSE.txt

from diffusers import ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
from torch import Generator
import cv2
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import os
import time
from tstitch_sd_utils import get_tstitch_pipepline_sdxl_controlnet

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    
    cache_dir = './pretrained_models'
    save_dir = f'./figures/controlnet_depth'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    prompt = "a beautiful bosai tree, masterpiece, 4K"
    negative_prompt = "blur, low quality, bad quality, sketches"


    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    controlnet_conditioning_scale = 0.5
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    ).to("cuda")

    image = load_image("./demo/bonsai.jpg")
    depth_image = get_depth_map(image)
    depth_image.save(f"{save_dir}/depth_image.png")


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
                        image=depth_image.resize((1024, 1024)), unet_s_ratio=ratio,
                        negative_prompt=negative_prompt).images[0]
        tic2 = time.time()
        print(f'ratio = {ratio}, time cost = {round(tic2 - tic1, 1)}s')
        image.save(f"{save_dir}/sample-ratio-{ratio}.png")

    with open(f'{save_dir}/prompt.txt', 'w') as f:
        f.write(prompt + '\n')
        f.write(negative_prompt + '\n')

