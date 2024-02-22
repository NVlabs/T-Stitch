# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under CC-BY-NC-SA-4.0.
# To view a copy of this license, visit 
# https://github.com/NVlabs/T-Stitch/blob/main/LICENSE.txt


"""
Sample new images from a pre-trained DiT.
"""
import json
import os.path

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import time


def two_models_combo(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8

    dit_names = ['DiT-S/2', 'DiT-XL/2']
    ckpt_paths = [args.s_ckpt, args.xl_ckpt]
    denoisers = []
    for i, name in enumerate(dit_names):
        print(f'loading {name} ...')
        dit_m = DiT_models[name](
            input_size=latent_size,
            num_classes=args.num_classes
        )
        dit_m.cfg_scale = args.cfg_scale
        state_dict = find_model(ckpt_paths[i])
        dit_m.load_state_dict(state_dict)
        denoisers.append(dit_m)

    from models import TStitch
    model = TStitch(denoisers, ratio=args.ratio).to(device)


    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    if args.solver == 'ddpm':
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
    elif args.solver == 'ddim':
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
    else:
        pass

    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_dir = f'./figures/seed_{args.seed}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_image(samples, f"{save_dir}/sample-{args.cfg_scale}-{args.solver}-steps-{args.num_sampling_steps}-seed-{args.seed}-ratio-{args.ratio}-s-b.png", nrow=4, normalize=True, value_range=(-1, 1))

def three_models_combo(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8

    dit_names = ['DiT-S/2', 'DiT-B/2', 'DiT-XL/2']
    ckpt_paths = [args.s_ckpt, args.b_ckpt, args.xl_ckpt]
    denoisers = []
    for i, name in enumerate(dit_names):
        print(f'loading {name} ...')
        dit_m = DiT_models[name](
            input_size=latent_size,
            num_classes=args.num_classes
        )
        dit_m.cfg_scale = args.cfg_scale
        state_dict = find_model(ckpt_paths[i])
        dit_m.load_state_dict(state_dict)
        denoisers.append(dit_m)

    from models import MultiTStitch
    model = MultiTStitch(denoisers).to(device)

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    model.fractions = args.three_combo

    s_frac, b_frac, xl_frac = args.three_combo

    # Sample images:
    if args.solver == 'ddpm':
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
    elif args.solver == 'ddim':
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
    else:
        pass

    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    save_dir = f'./figures/seed_{args.seed}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_image(samples, f"{save_dir}/sample-{args.cfg_scale}-{args.solver}-steps-{args.num_sampling_steps}-seed-{args.seed}-s-{s_frac}-b-{b_frac}-xl-{xl_frac}.png", nrow=4, normalize=True, value_range=(-1, 1))


def two_models_combo_all_tradeoffs(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8


    dit_names = ['DiT-S/2', 'DiT-XL/2']
    ckpt_paths = [args.s_ckpt, args.xl_ckpt]
    denoisers = []
    for i, name in enumerate(dit_names):
        print(f'loading {name} ...')
        dit_m = DiT_models[name](
            input_size=latent_size,
            num_classes=args.num_classes
        )
        dit_m.cfg_scale = args.cfg_scale
        state_dict = find_model(ckpt_paths[i])
        dit_m.load_state_dict(state_dict)
        denoisers.append(dit_m)

    from models import TStitch
    model = TStitch(denoisers, ratio=args.ratio).to(device)

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    class_labels = [207]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    small_candidates = [round(item, 1) for item in np.arange(0., 1.1, 0.1).tolist()]
    all_images = []
    for small_ratio in small_candidates:
        print(f"Fraction of DiT-S: {small_ratio}")
        model.ratio = small_ratio


        # Sample images:
        if args.solver == 'ddpm':
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                device=device
            )
        elif args.solver == 'ddim':
            samples = diffusion.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                device=device
            )
        else:
            pass

        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample
        all_images.append(samples)

    all_images = torch.stack(all_images, dim=0).permute(1, 0, 2, 3, 4)
    # Save and display images:
    save_dir = f'./figures/seed_{args.seed}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    for i, image in enumerate(all_images):
        save_image(image, f"{save_dir}/class-{class_labels[i]}.png", nrow=len(small_candidates), normalize=True, value_range=(-1, 1))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solver", type=str, default="ddim")
    parser.add_argument("--save_dir", type=str, default="./samples")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--s_ckpt", type=str, default='pretrained_models/dit_s_256.pt')
    parser.add_argument("--b_ckpt", type=str, default='pretrained_models/dit_b_256.pt')
    parser.add_argument("--xl_ckpt", type=str, default='pretrained_models/dit_xl_256.pt')
    parser.add_argument("--all_tradeoffs", action='store_true')
    parser.add_argument('--three_combo', 
                        metavar='float', 
                        type=float, 
                        default=None,
                        nargs='+',
                        help='A list of float numbers separated by spaces')

    args = parser.parse_args()
    if args.all_tradeoffs:
        two_models_combo_all_tradeoffs(args)
    elif args.three_combo is not None:
        three_models_combo(args)
    else:
        two_models_combo(args)
