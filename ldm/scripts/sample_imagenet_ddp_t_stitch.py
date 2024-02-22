# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under CC-BY-NC-SA-4.0.
# To view a copy of this license, visit 
# https://github.com/NVlabs/T-Stitch/blob/main/LICENSE.txt


import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
# import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from tqdm import tqdm
import os
import math

from ldm.models.diffusion.ddim import DDIMSampler
import torch.distributed as dist
import argparse


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    # os.system('rm -rf ' + sample_dir)
    return npz_path

def load_model_from_config(configs, ckpts, ae_ckpt):

    model = instantiate_from_config(configs.model)

    auto_encoder_ckpt = torch.load(ae_ckpt)
    m, u = model.first_stage_model.load_state_dict(auto_encoder_ckpt['state_dict'], strict=False)
    print('autoencoder missing keys: ', m)
    print('autoencoder unexpected keys: ', u)

    target_dict = {}

    for i, ckpt in enumerate(ckpts):
        pl_sd = torch.load(ckpt)
        sd = pl_sd["state_dict"]

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


def get_model(cfg, ckpt_path, ae_ckpt):
    config = OmegaConf.load(cfg)
    model = load_model_from_config(config, ckpt_path, ae_ckpt)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-sampling-steps", type=int, default=200)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ldm_s_path", type=str, default='./pretrained_models/ldm_s.ckpt')
    parser.add_argument("--ldm_path", type=str, default='./pretrained_models/ldm.ckpt')
    parser.add_argument("--ae_ckpt", type=str, default="./pretrained_models/vq-f8/model.ckpt")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--cfg", type=str, default='configs/latent-diffusion/cin-ldm-vq-f8-t-stitch.yaml')
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--ddim_eta", type=int, default=1.0)

    args = parser.parse_args()


    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    folder_name = f"seed-{args.global_seed}-steps-{args.num_sampling_steps}-images-{args.num_fid_samples}-ratio-{args.ratio}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"


    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    ckpts = [
        args.ldm_s_path,
        args.ldm_path
    ]

    model = get_model(args.cfg, ckpts, args.ae_ckpt)
    sampler = DDIMSampler(model)

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    latent_size = 32
    ddim_eta = 1.0
    model.model.diffusion_model.ratio = args.ratio

    for _ in pbar:
        y = torch.randint(0, args.num_classes, (n,), device=device)
        c = model.get_learned_conditioning({model.cond_stage_key: y})
        uc = []
        for temp_c in c:
            uc.append(torch.rand_like(temp_c, device=device))
        samples_ddim, _ = sampler.sample(S=args.num_sampling_steps,
                                         conditioning=c,
                                         batch_size=n,
                                         shape=[4, 32, 32],
                                         verbose=False,
                                         unconditional_guidance_scale=args.cfg_scale,
                                         unconditional_conditioning=uc,
                                         eta=ddim_eta, progress_bar=False)

        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                     min=0.0, max=1.0)

        x_samples_ddim = (x_samples_ddim * 255.).to("cpu", dtype=torch.uint8).permute(0, 2, 3, 1).numpy()
        for i, sample in enumerate(x_samples_ddim):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()