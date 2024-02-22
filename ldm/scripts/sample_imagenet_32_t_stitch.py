# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under CC-BY-NC-SA-4.0.
# To view a copy of this license, visit 
# https://github.com/NVlabs/T-Stitch/blob/main/LICENSE.txt


import os.path
import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
# from taming.models import vqgan
#@title loading utils
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import argparse


import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.utils import make_grid, save_image

import random

def generate_random_filename(length=10):
  """Generates a random filename of the given length.

  Args:
    length: The length of the filename.

  Returns:
    A random filename.
  """

  chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
  filename = ''.join(random.choice(chars) for _ in range(length))
  return filename


def load_model_from_config(configs, ckpts, ae_ckpt=None):
    model = instantiate_from_config(configs.model)
    auto_encoder_ckpt = torch.load(ae_ckpt)
    m, u = model.first_stage_model.load_state_dict(auto_encoder_ckpt['state_dict'], strict=False)
    print('autoencoder missing keys: ', m)
    print('autoencoder unexpected keys: ', u)


    target_dict = {}


    for i, ckpt in enumerate(ckpts):
        pl_sd = torch.load(ckpt)
        sd = pl_sd["state_dict"]
        # continue
        for name, data in sd.items():
            new_name = name
            if 'diffusion_model' in name:
                if name.startswith('model_ema'):
                    new_name = new_name.replace('diffusion_model', 'diffusion_modelanchors' + str(i))
                else:
                    new_name = new_name.replace('diffusion_model', 'diffusion_model.anchors.' + str(i))

            if 'cond_stage_model' in name:
                if name.startswith('model_ema'):
                    new_name = new_name.replace('cond_stage_model', 'cond_stage_model' + str(i))
                else:
                    new_name = new_name.replace('cond_stage_model', 'cond_stage_model.' + str(i))
            # print(new_name)
            target_dict[new_name] = data
    m, u = model.load_state_dict(target_dict, strict=False)
    print('diffusion missing keys: ', m)
    print('diffusion unexpected keys: ', u)
    model.cuda()
    model.eval()
    return model


def get_model(config_path, ckpt_path, auto_encoder_path=None):
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path, auto_encoder_path)
    return model


def gen_images(args):
    ckpts = [
        args.ldm_s_path,
        args.ldm_path
    ]
    model = get_model(args.cfg, ckpts, args.ae_ckpt)
    sampler = DDIMSampler(model)

    classes = [25, 187, 448, 992]  # define classes to be sampled here
    n_samples_per_class = 8

    ddim_steps = 100
    ddim_eta = 1.0
    scale = 3.0  # for unconditional guidance
    model.model.diffusion_model.ratio = args.ratio

    # warmup
    all_samples = list()
    with torch.no_grad():
        with model.ema_scope():
            for class_label in classes:
                print(
                    f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class * [class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                uc = []
                for temp_c in c:
                    uc.append(torch.rand_like(temp_c, device=model.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples_per_class,
                                                 shape=[4, 32, 32],
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                             min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)

    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    save_dir = f'figures/seed-{args.seed}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    Image.fromarray(grid.astype(np.uint8)).save(f"{save_dir}/samples-faction-{args.ratio}.png")


def gen_images_all_ratios(args):
    ckpts = [
        args.ldm_s_path,
        args.ldm_path
    ]
    model = get_model(args.cfg, ckpts, args.ae_ckpt)
    sampler = DDIMSampler(model)
    # torch.manual_seed(args.seed)

    classes = [25, 187, 448, 992]

    ddim_steps = args.sampling_steps
    ddim_eta = 1.0
    scale = args.cfg_scale  # for unconditional guidance


    # warmup
    all_samples = list()

    all_ratios = np.arange(0, 1.1, 0.1).tolist()
    all_ratios = [round(ratio, 1) for ratio in all_ratios]

    # n = 8
    x_T = torch.randn(len(classes), 4, 32, 32).cuda()
    xc = torch.tensor(classes)
    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

    uc1 = torch.rand_like(c[0], device=model.device)
    target_shape = c[1].shape[-1]
    uc2 = torch.nn.functional.interpolate(uc1, target_shape)
    uc = [uc1, uc2]

    with torch.no_grad():
        with model.ema_scope():
            for ratio in all_ratios:
                print(f'ratio: {ratio}')
                model.model.diffusion_model.ratio = ratio
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=len(classes),
                                                 shape=[4, 32, 32],
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta, x_T=x_T)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                all_samples.append(x_samples_ddim)


    # display as grid
    grid = torch.stack(all_samples, 0).permute(1, 0, 2, 3, 4)
    save_dir = f'figures/seed-{args.seed}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, sample in enumerate(grid):
        random_filename = generate_random_filename(5)
        save_image(sample, f'{save_dir}/sample-all-ratios-class-{classes[i]}-{random_filename}.png', nrow=len(all_ratios), normalize=True, value_range=(-1, 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.4)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--ae_ckpt", type=str, default='./pretrained_models/vq-f8/model.ckpt')
    parser.add_argument("--ldm_s_path", type=str, default='./pretrained_models/ldm_s.ckpt')
    parser.add_argument("--ldm_path", type=str, default='./pretrained_models/ldm.ckpt')
    parser.add_argument("--cfg", type=str, default='configs/latent-diffusion/cin-ldm-vq-f8-t-stitch.yaml')
    parser.add_argument("--all_ratios", action='store_true')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    if args.all_ratios:
        gen_images_all_ratios(args)
    else:
        gen_images(args)